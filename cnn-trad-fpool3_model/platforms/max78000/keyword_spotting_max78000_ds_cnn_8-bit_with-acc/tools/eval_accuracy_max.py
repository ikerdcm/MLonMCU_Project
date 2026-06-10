#!/usr/bin/env python3
"""Unified on-device accuracy eval for the DS-CNN (device-in-the-loop).

Streams real Google Speech Commands *test* audio to the board, which runs its
OWN frontend (ds_cnn_frontend_compute) + model and returns a prediction. The
host tallies accuracy + a 12-class confusion matrix. Because the common input is
RAW AUDIO (not MFCC), the same test set works on every board despite each model
being scale-locked to its own frontend's MFCC — and it measures the true
deployed pipeline (frontend + model + quantization), so it catches frontend bugs
that an MFCC-only eval would hide. (MAX-first; the protocol is board-agnostic.)

Serial protocol (host <-> board, 115200), half-duplex per clip:
  host -> board : "EVAL <idx> <nsamples>\n"  then  nsamples * int16 little-endian
  board -> host : "BENCH,event=eval,idx=<idx>,pred_idx=<p>\r\n"
The board must boot in EVAL mode (KWS20_CFG_APP_MODE_KWS_EVAL).

Usage:
  python3 tools/eval_accuracy_max.py --per-class 10 --port /dev/tty.usbmodem11302
  python3 tools/eval_accuracy_max.py --per-class 5 --no-serial   # build+check subset only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# DS-CNN 12-class order — MUST match the firmware `labels[]` / protocol.LABELS.
LABELS = ["down", "go", "left", "no", "off", "on",
          "right", "stop", "up", "yes", "silence", "unknown"]
# Google Speech Commands raw_test/ folder -> our class index.
GSC_DIR_TO_IDX = {
    "down": 0, "go": 1, "left": 2, "no": 3, "off": 4, "on": 5,
    "right": 6, "stop": 7, "up": 8, "yes": 9, "_silence_": 10, "_unknown_": 11,
}
WINDOW = 16000  # 1 s @ 16 kHz, the model's audio window

DEFAULT_DATASET = Path("/Users/iker/MAX78000_Toolchain/ai8x-training/data/KWS/raw_test")
HERE = Path(__file__).resolve().parent


def load_clip(path: Path) -> np.ndarray:
    """Load a 16 kHz mono wav -> int16[WINDOW] (zero-padded / truncated)."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.int16)
    if len(data) < WINDOW:
        data = np.concatenate([data, np.zeros(WINDOW - len(data), np.int16)])
    return data[:WINDOW]


def build_subset(dataset: Path, per_class: int, seed: int):
    """Deterministic stratified subset: per_class clips per available class."""
    rng = np.random.default_rng(seed)
    items = []  # (wav_path, true_idx)
    for d, idx in sorted(GSC_DIR_TO_IDX.items(), key=lambda kv: kv[1]):
        wavs = sorted((dataset / d).glob("*.wav"))
        if not wavs:
            print(f"  ! no wavs for '{d}' (idx {idx}) — skipping")
            continue
        pick = [wavs[i] for i in rng.choice(len(wavs), min(per_class, len(wavs)), replace=False)]
        items += [(p, idx) for p in sorted(pick)]
    return items


def confusion_and_metrics(pairs):
    """pairs = [(true, pred)] -> (12x12 matrix, accuracy, per-class P/R)."""
    n = len(LABELS)
    cm = [[0] * n for _ in range(n)]
    for t, p in pairs:
        if 0 <= t < n and 0 <= p < n:
            cm[t][p] += 1
    correct = sum(cm[i][i] for i in range(n))
    total = sum(sum(r) for r in cm)
    acc = correct / total if total else 0.0
    per_class = {}
    for i, lab in enumerate(LABELS):
        tp = cm[i][i]
        support = sum(cm[i])
        pred_i = sum(cm[r][i] for r in range(n))
        per_class[lab] = {
            "support": support,
            "recall": tp / support if support else None,
            "precision": tp / pred_i if pred_i else None,
        }
    return cm, acc, per_class


def print_confusion(cm):
    print("\nConfusion matrix (rows = true, cols = pred):")
    print("       " + " ".join(f"{l[:4]:>4}" for l in LABELS))
    for i, lab in enumerate(LABELS):
        print(f"{lab[:6]:>6} " + " ".join(f"{cm[i][j]:>4}" for j in range(len(LABELS))))


def stream_eval(items, port, baud, timeout):
    """Stream each clip, collect predictions. Returns [(true, pred)]."""
    import serial
    pairs = []
    with serial.Serial(port, baud, timeout=timeout) as ser:
        time.sleep(0.3)
        ser.reset_input_buffer()
        for idx, (path, true_idx) in enumerate(items):
            audio = load_clip(path)
            ser.write(f"EVAL {idx} {WINDOW}\n".encode())
            ser.write(audio.astype("<i2").tobytes())   # int16 little-endian
            ser.flush()
            pred = None
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = ser.readline().decode("utf-8", "replace").strip()
                if line.startswith("BENCH,event=eval") and f"idx={idx}" in line:
                    for kv in line.split(","):
                        if kv.startswith("pred_idx="):
                            try:
                                pred = int(kv.split("=", 1)[1])
                            except ValueError:
                                pred = None
                    break
            ok = "ok" if pred == true_idx else "X "
            print(f"  [{idx+1}/{len(items)}] {ok} true={LABELS[true_idx]:8} "
                  f"pred={LABELS[pred] if pred is not None and 0 <= pred < len(LABELS) else pred}")
            if pred is None:
                print("    ! no prediction (timeout) — is the board in EVAL mode?")
            pairs.append((true_idx, pred if pred is not None else -1))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--no-serial", action="store_true",
                    help="build + sanity-check the subset only (no board)")
    ap.add_argument("--out", type=Path, default=HERE.parent / "measurements" / "accuracy")
    args = ap.parse_args()

    if not args.dataset.exists():
        sys.exit(f"dataset not found: {args.dataset} (pass --dataset)")

    print(f"Building subset: {args.per_class}/class from {args.dataset}")
    items = build_subset(args.dataset, args.per_class, args.seed)
    print(f"  {len(items)} clips across {len({i for _, i in items})} classes")

    if args.no_serial:
        a = load_clip(items[0][0])
        print(f"  sanity: first clip {items[0][0].name} -> int16[{len(a)}] "
              f"range [{a.min()},{a.max()}] (true={LABELS[items[0][1]]})")
        print("  (--no-serial: subset OK; rerun with --port to evaluate on the board)")
        return

    port = args.port
    if not port:
        import glob
        hits = sorted(glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/cu.usbmodem*"))
        port = hits[0] if hits else None
    if not port:
        sys.exit("no serial port — pass --port")

    print(f"Evaluating on {port} (board must be in EVAL mode)...")
    pairs = stream_eval(items, port, args.baud, args.timeout)
    cm, acc, per_class = confusion_and_metrics(pairs)
    print_confusion(cm)
    print(f"\nAccuracy: {acc*100:.2f}%  ({sum(cm[i][i] for i in range(len(LABELS)))}/{len(items)})")

    out_dir = args.out / datetime.now().strftime("eval_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "board": "max78000_ds_cnn_v1",
        "dataset": str(args.dataset),
        "per_class": args.per_class, "seed": args.seed,
        "n_clips": len(items), "accuracy": acc,
        "labels": LABELS, "confusion_matrix": cm, "per_class": per_class,
    }, indent=2))
    print(f"\nWrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
