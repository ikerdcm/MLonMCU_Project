"""Build + flash orchestration for the dashboard.

Routes all flashing through platforms/flash_all.sh, which already knows each
MCU's mode mechanism (Coral script choice; MAX/STM32 kws20_mode_config.h edit)
and targets each board via its own debug probe (no serial-port ambiguity).
"""
from __future__ import annotations

from pathlib import Path

# mcu_stream/ -> dashboard/ -> platforms/
PLATFORMS_DIR = Path(__file__).resolve().parents[2]
FLASH_ALL = PLATFORMS_DIR / "flash_all.sh"

# Dashboard MCU key -> flash_all.sh --only token
ONLY_TOKEN = {"coral": "coral", "stm32u5": "stm32", "max78000": "max"}

# Modes flash_all.sh understands. Coral "bench" maps to offline there.
FLASH_MODES = ("live", "offline")

# Report-class gating — applies to all 3 boards. label -> flash_all.sh --report token.
REPORT_MODES = {
    "Keywords only": "keywords",
    "+ unknown": "unknown",
    "+ silence": "silence",
    "+ unknown & silence": "both",
}


# Per-MCU flashable models: label -> flash_all.sh --max-model token.
# Only MAX has alternates today; the others ship a single model.
MODELS = {
    "max78000": {"DS-CNN (INT8+accel)": "dscnn",
                 "kws20_demo": "kws20_demo",
                 "kws20_demo_w90": "kws20_demo_w90"},
    "stm32u5":  {"DS-CNN (INT8)": "dscnn"},
    "coral":    {"DS-CNN": "dscnn"},
}

# Which label set each model's pred_idx uses (see protocol.LABEL_SETS).
MODEL_LABEL_SET = {
    "dscnn": "dscnn", "kws20_demo": "kws20", "kws20_demo_w90": "kws20",
}


def flash_command(mode: str, mcu: str | None = None,
                  report: str = "keywords", max_model: str = "dscnn") -> list[str]:
    """Build the flash_all.sh command. mcu=None flashes all three."""
    cmd = ["bash", str(FLASH_ALL), "--mode", mode, "--report", report,
           "--max-model", max_model]
    if mcu is not None:
        cmd += ["--only", ONLY_TOKEN[mcu]]
    return cmd


def flash_available() -> bool:
    return FLASH_ALL.exists()
