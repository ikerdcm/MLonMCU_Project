"""Shared serial protocol parsing for the MCU dashboard.

All three MCUs (Coral, STM32U5, MAX78000) stream keyword-spotting inference
results over a UART/USB-CDC serial port at 115200 baud. Two line formats exist:

1. BENCH CSV  (STM32 both modes, MAX78000 both modes, Coral `bench` mode):
       BENCH,event=inference,run=3,mode=live,cnn_us=1234,cycles=98720,pred_idx=2
   Other events: model_info, acquisition, done, *_error.

2. Coral `live` pretty format:
       >>> left     (73%)  mfcc=112345 us  infer=1234 us
           unknown  ( 5%)  mfcc=112345 us  infer=1234 us
   (leading ">>>" = confident.)

This module turns either format into a single normalized `InferenceRecord`,
so the GUI never has to care which MCU produced a line.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

# DS-CNN-L 12-class label order — identical across all three firmwares.
LABELS = [
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown",
]


def label_from_idx(idx: Optional[int]) -> Optional[str]:
    if idx is None:
        return None
    if 0 <= idx < len(LABELS):
        return LABELS[idx]
    return f"idx{idx}"


# MAX78000 kws20_demo is a 21-class model with a different label order/size than
# the 12-class DS-CNN. The dashboard must decode pred_idx with the matching set.
KWS20_LABELS = [
    "up", "down", "left", "right", "stop", "go", "yes", "no", "on", "off",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "zero", "unknown",
]

LABEL_SETS = {"dscnn": LABELS, "kws20": KWS20_LABELS}


def label_in_set(idx: Optional[int], set_name: str) -> Optional[str]:
    """Decode pred_idx using the named label set (falls back to DS-CNN)."""
    if idx is None:
        return None
    labels = LABEL_SETS.get(set_name, LABELS)
    return labels[idx] if 0 <= idx < len(labels) else f"idx{idx}"


# ── MCU profiles ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class McuProfile:
    key: str            # internal id
    name: str           # display name
    modes: tuple        # selectable firmware modes
    usb_vid: int = 0    # USB vendor id of this board's debug/CDC interface
    default_baud: int = 115200
    assert_dtr: bool = False   # Coral CDC drops output until DTR is asserted
    port_hint: str = "usbmodem"


PROFILES = {
    "coral": McuProfile(
        key="coral", name="Coral Dev Board Micro",
        modes=("live", "bench"), usb_vid=0x18d1, assert_dtr=True,
    ),
    "stm32u5": McuProfile(
        key="stm32u5", name="STM32U5",
        modes=("live", "offline"), usb_vid=0x0483,  # ST-Link
    ),
    "max78000": McuProfile(
        key="max78000", name="MAX78000",
        modes=("live", "offline"), usb_vid=0x0d28,  # DAPLink CMSIS-DAP
    ),
}


# ── Normalized record ───────────────────────────────────────────────────────
@dataclass
class InferenceRecord:
    mcu: str                       # profile key
    mode: str                      # firmware mode
    t_host: float                  # host monotonic clock (for cross-MCU sync)
    t_wall: float                  # epoch seconds (for display)
    event: str                     # inference | model_info | acquisition | done | error | other
    raw: str
    run: Optional[int] = None
    pred_idx: Optional[int] = None
    label: Optional[str] = None
    confidence: Optional[float] = None   # percent, only Coral live provides it
    infer_us: Optional[int] = None       # unified cnn_us / infer_us
    cycles: Optional[int] = None
    level: Optional[int] = None          # mic audio level/RMS (event=level)
    thr: Optional[int] = None            # adaptive voice-trigger threshold (event=level)
    scores: Optional[list] = None        # per-class scores 0..100 (event=scores)
    mfcc: Optional[list] = None          # flattened MFCC int8 values (event=mfcc)
    mfcc_w: Optional[int] = None         # MFCC width (bins) / height (frames)
    mfcc_h: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def is_inference(self) -> bool:
        return self.event == "inference"


# ── Low-level parsers ───────────────────────────────────────────────────────
def parse_bench(line: str) -> Optional[dict]:
    """Parse a `BENCH,key=value,...` line into a dict, or None."""
    if not line.startswith("BENCH,"):
        return None
    fields = {}
    for part in line.split(",")[1:]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        fields[k.strip()] = v.strip()
    return fields


_CORAL_LIVE_RE = re.compile(
    r"^(?P<conf>>>>|)\s*(?P<label>\w+)\s+\(\s*(?P<score>\d+)%\)\s+"
    r"mfcc=(?P<mfcc>\d+)\s+us\s+infer=(?P<infer>\d+)\s+us"
)


def parse_coral_live(line: str) -> Optional[dict]:
    m = _CORAL_LIVE_RE.match(line.strip())
    if not m:
        return None
    return {
        "confident": m.group("conf") == ">>>",
        "label": m.group("label"),
        "score": int(m.group("score")),
        "mfcc_us": int(m.group("mfcc")),
        "infer_us": int(m.group("infer")),
    }


def _to_int(d: dict, *keys) -> Optional[int]:
    for k in keys:
        if k in d:
            try:
                return int(d[k])
            except (ValueError, TypeError):
                return None
    return None


# ── Top-level: line -> normalized record ────────────────────────────────────
def build_record(mcu: str, mode: str, line: str,
                 t_host: Optional[float] = None,
                 t_wall: Optional[float] = None) -> Optional[InferenceRecord]:
    """Turn one raw serial line into a normalized record.

    Returns None for noise (banners, blank lines) that carry no event.
    """
    if t_host is None:
        t_host = time.monotonic()
    if t_wall is None:
        t_wall = time.time()
    line = line.rstrip("\r\n")
    if not line.strip():
        return None

    # 1. BENCH CSV (every MCU/mode except Coral live)
    fields = parse_bench(line)
    if fields is not None:
        event = fields.get("event", "other")
        idx = _to_int(fields, "pred_idx")

        def _ilist(key):  # ";"-separated ints (commas are the field separator)
            if key not in fields:
                return None
            try:
                return [int(x) for x in fields[key].split(";") if x != ""]
            except (ValueError, TypeError):
                return None

        rec = InferenceRecord(
            mcu=mcu, mode=mode, t_host=t_host, t_wall=t_wall,
            event=event, raw=line,
            run=_to_int(fields, "run"),
            pred_idx=idx,
            label=label_from_idx(idx),
            infer_us=_to_int(fields, "cnn_us", "infer_us"),
            cycles=_to_int(fields, "cycles"),
            level=_to_int(fields, "rms", "level"),
            thr=_to_int(fields, "thr"),
            scores=_ilist("s"),
            mfcc=_ilist("d"),
            mfcc_w=_to_int(fields, "w"),
            mfcc_h=_to_int(fields, "h"),
            extra={k: v for k, v in fields.items()
                   if k not in ("event", "run", "pred_idx", "cnn_us",
                                "infer_us", "cycles", "rms", "level", "thr",
                                "s", "d", "w", "h")},
        )
        return rec

    # 2. Coral live pretty format
    if mcu == "coral":
        live = parse_coral_live(line)
        if live is not None:
            label = live["label"]
            idx = LABELS.index(label) if label in LABELS else None
            return InferenceRecord(
                mcu=mcu, mode=mode, t_host=t_host, t_wall=t_wall,
                event="inference", raw=line,
                pred_idx=idx, label=label,
                confidence=float(live["score"]),
                infer_us=live["infer_us"],
                extra={"mfcc_us": live["mfcc_us"],
                       "confident": live["confident"]},
            )

    # 3. Unrecognized (boot banner, log noise) — not an event.
    return None
