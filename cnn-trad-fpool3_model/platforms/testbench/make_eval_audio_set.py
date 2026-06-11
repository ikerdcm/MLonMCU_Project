#!/usr/bin/env python3
"""Bake a GSC test subset as RAW AUDIO into Coral's LittleFS eval file.

Coral's console drops host->device USB bytes, so we can't stream audio to it like
MAX/U5. Instead we bake the subset into /eval/audio_set.bin (copied into LittleFS
at flash time); the kws_eval_stream[_cpu] app reads it, runs the frontend + model
on-device, and prints pred + the embedded true label for testbench.py to tally.

Format:  uint32 N  |  N*16000 int16 LE audio  |  N uint8 labels (our 12-class idx)

Usage:  python3 make_eval_audio_set.py --per-class 12   # then build+flash the app
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

from testbench import build_subset, load_clip, WINDOW, DEFAULT_DATASET, LABELS

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[0] / "coral" / "eval" / "audio_set.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.dataset.exists():
        sys.exit(f"dataset not found: {args.dataset}")

    items = build_subset(args.dataset, args.per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", len(items)))
        for path, _ in items:
            f.write(load_clip(path).astype("<i2").tobytes())
        f.write(bytes(idx for _, idx in items))

    mb = args.out.stat().st_size / 1e6
    per = {LABELS[i]: sum(1 for _, j in items if j == i) for i in range(12)}
    print(f"wrote {args.out}")
    print(f"  N={len(items)}  ({mb:.2f} MB)  per-class={per}")
    print("  next: build+flash the Coral eval app (it bakes this into LittleFS)")


if __name__ == "__main__":
    main()
