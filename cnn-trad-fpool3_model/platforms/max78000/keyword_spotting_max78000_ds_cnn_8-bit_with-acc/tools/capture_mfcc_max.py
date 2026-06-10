#!/usr/bin/env python3
"""Capture the live MFCC the MAX78000 DS-CNN v1 model actually sees and compare
it to the known-good reference (ds_cnn_test_input_left.h).

WHY: offline/bench feeds the model a perfect training-pipeline MFCC, so it only
proves the model is fine. Live computes MFCC on-device via ds_cnn_frontend_compute
— the only thing that differs. This reads the on-device int8 MFCC (emitted as
BENCH,event=mfcc,...,d=...; needs the firmware build that scales by *127) for each
spoken word, and prints per-coefficient stats + saturation so we can see whether
the live frontend produces model-compatible MFCC or not.

Usage:
  # disconnect the dashboard first (it holds the serial port)
  python3 tools/capture_mfcc_max.py --port /dev/tty.usbmodemXXXX --n 4
  python3 tools/capture_mfcc_max.py            # autodetect port, capture 3

Then speak a keyword (e.g. "left", which matches the reference) per inference.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

try:
    import serial  # pyserial (same dep the dashboard uses)
except ImportError:
    sys.exit("pyserial not found — run with the dashboard's interpreter, or: pip install pyserial")

LABELS = ["down", "go", "left", "no", "off", "on",
          "right", "stop", "up", "yes", "silence", "unknown"]
N_FRAMES, N_COEFF = 49, 10
HERE = Path(__file__).resolve().parent
REF_HEADER = HERE.parent / "ds_cnn_test_input_left.h"


def load_reference_int8():
    """Parse ds_cnn_test_input_left.h → 490 floats in [-1,1] → int8 (*127)."""
    if not REF_HEADER.exists():
        return None
    text = REF_HEADER.read_text()
    body = text[text.index("{") + 1: text.rindex("}")]
    vals = [float(x) for x in re.findall(r"-?\d+\.\d+e[+-]\d+|-?\d+\.\d+|-?\d+", body)]
    vals = vals[: N_FRAMES * N_COEFF]
    if len(vals) < N_FRAMES * N_COEFF:
        return None
    return [max(-128, min(127, round(v * 127))) for v in vals]


def per_coeff_stats(flat):
    """flat = 490 ints (49 frames x 10 coeffs, row-major) -> per-coeff (min,mean,max)."""
    cols = [[flat[f * N_COEFF + c] for f in range(N_FRAMES)] for c in range(N_COEFF)]
    return [(min(col), sum(col) / len(col), max(col)) for col in cols]


def saturation_pct(flat):
    sat = sum(1 for v in flat if v <= -127 or v >= 127)
    return 100.0 * sat / len(flat)


def print_block(title, flat, pred=None):
    print(f"\n=== {title} ===")
    if pred is not None:
        lbl = LABELS[pred] if 0 <= pred < len(LABELS) else f"idx{pred}"
        print(f"  predicted: {lbl} (idx {pred})")
    print(f"  saturation (|v|>=127): {saturation_pct(flat):.1f}% of 490")
    print("  coeff |  min   mean   max")
    for c, (lo, mu, hi) in enumerate(per_coeff_stats(flat)):
        print(f"   {c:>4d} | {lo:>5d} {mu:>6.1f} {hi:>5d}")


def autodetect_port():
    for pat in ("/dev/tty.usbmodem*", "/dev/cu.usbmodem*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=3, help="inferences to capture")
    args = ap.parse_args()

    port = args.port or autodetect_port()
    if not port:
        sys.exit("no serial port found — pass --port /dev/tty.usbmodemXXXX")

    ref = load_reference_int8()
    if ref:
        print_block("REFERENCE (ds_cnn_test_input_left, 'left', *127)", ref)
    else:
        print("(reference header not parsed — showing live only)")

    print(f"\nListening on {port} @ {args.baud} — speak a keyword per inference "
          f"(capturing {args.n})...\n")
    last_pred = None
    got = 0
    with serial.Serial(port, args.baud, timeout=2) as ser:
        while got < args.n:
            raw = ser.readline().decode("utf-8", "replace").strip()
            if not raw.startswith("BENCH,"):
                continue
            fields = dict(p.split("=", 1) for p in raw.split(",")[1:] if "=" in p)
            ev = fields.get("event")
            if ev == "inference":
                try:
                    last_pred = int(fields["pred_idx"])
                except (KeyError, ValueError):
                    last_pred = None
            elif ev == "mfcc" and "d" in fields:
                try:
                    flat = [int(x) for x in fields["d"].split(";") if x != ""]
                except ValueError:
                    continue
                if len(flat) != N_FRAMES * N_COEFF:
                    print(f"(skipped mfcc with {len(flat)} values)")
                    continue
                got += 1
                print_block(f"LIVE #{got}", flat, last_pred)

    print("\nInterpretation hints:")
    print("  - coeff 0 ~ -127 (clamped) is expected (energy term), like the reference.")
    print("  - coeffs 1-9 should be modest (~ +/-5..25) like the reference. If they")
    print("    SATURATE (many at +/-127) -> live MFCC too large (scale/pre-emph/power).")
    print("    If they're ~0/flat -> audio too quiet or frontend underflow.")


if __name__ == "__main__":
    main()
