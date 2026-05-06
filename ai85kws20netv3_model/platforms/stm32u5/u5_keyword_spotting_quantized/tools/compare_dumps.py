#!/usr/bin/env python3
"""
Compare an STM32U5 live AI input dump against a MAX78000 reference dump.

Both inputs are 16384-sample int8 captures of what was fed to the keyword-
spotting model.  The U5 model itself is verified to be correct (TEST 2 in
the offline harness PASSes the MAX78000 vector), so any prediction gap
must come from differences in the audio captured by U5 vs MAX78000.

This script extracts both dumps, then plots / prints:
  * Waveform side-by-side (time domain)
  * Power spectrum (FFT in dB) and the U5-minus-MAX delta spectrum
  * Histogram comparison
  * Spectrogram
  * Numeric summary (peak, RMS, zero count, band energies)

Usage:
    python3 compare_dumps.py <u5_log_path> [--u5-dump-index N]
                             [--max-log <max_log_path>] [--max-dump-index N]
                             [--save <prefix>]

If --max-log is omitted, the bundled reference at
    ai85kws20netv3_model/test_vectors/max78000_ai_input_dump.log
is used.

The U5 log format is a serial capture from kws20_live.c which contains one or
more sections of the form:
    AI_INPUT_DUMP_BEGIN
    <comma-separated int8 values, 32 per line, trailing comma>
    AI_INPUT_DUMP_END

The MAX78000 reference log uses the same delimiters but its values are stored
as uint8 bytes (0..255); this script reinterprets them as signed int8.

Sample rate is assumed to be 16 kHz on both sides.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: pip3 install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    from scipy import signal as sp_signal
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

SAMPLE_RATE = 16000
EXPECTED_LEN = 16384

DEFAULT_MAX_LOG = (
    Path(__file__).resolve().parents[4]
    / "test_vectors"
    / "max78000_ai_input_dump.log"
)


def extract_dump(log_path: Path, dump_index: int = 0,
                 reinterpret_signed: bool = False) -> np.ndarray:
    """Pull the Nth AI_INPUT_DUMP_BEGIN..END block out of a serial log."""
    text = log_path.read_text(errors="replace")
    blocks = re.findall(
        r"AI_INPUT_DUMP_BEGIN\s*\n(.*?)\nAI_INPUT_DUMP_END",
        text, re.DOTALL,
    )
    if not blocks:
        raise RuntimeError(f"No AI_INPUT_DUMP_* blocks in {log_path}")
    if dump_index >= len(blocks):
        raise RuntimeError(
            f"{log_path} has {len(blocks)} dump(s); requested index {dump_index}"
        )

    raw = blocks[dump_index].replace("\n", "")
    nums = [int(x) for x in raw.split(",") if x.strip()]

    if len(nums) != EXPECTED_LEN:
        print(f"warning: {log_path}#{dump_index} has {len(nums)} samples, "
              f"expected {EXPECTED_LEN}", file=sys.stderr)

    arr = np.array(nums, dtype=np.int32)
    if reinterpret_signed:
        arr = np.where(arr > 127, arr - 256, arr)

    return arr.astype(np.float32)


def looks_uint8(arr: np.ndarray) -> bool:
    """Heuristic: MAX78000 dumps are uint8 bytes (0..255)."""
    return arr.min() >= 0 and arr.max() > 127


def histogram_bins(arr: np.ndarray) -> dict[str, int]:
    a = np.abs(arr).astype(np.int32)
    return {
        "0": int(np.sum(a == 0)),
        "1-2": int(np.sum((a >= 1) & (a <= 2))),
        "3-10": int(np.sum((a >= 3) & (a <= 10))),
        "11-20": int(np.sum((a >= 11) & (a <= 20))),
        "21-40": int(np.sum((a >= 21) & (a <= 40))),
        "41+": int(np.sum(a >= 41)),
    }


def band_energy_db(spec: np.ndarray, freqs: np.ndarray,
                   bands: list[tuple[float, float]]) -> list[float]:
    out = []
    eps = 1e-12
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            out.append(float("-inf"))
            continue
        e = float(np.sum(spec[mask] ** 2))
        out.append(10.0 * np.log10(e + eps))
    return out


def print_summary(name: str, x: np.ndarray) -> None:
    n = len(x)
    abs_x = np.abs(x)
    print(f"--- {name} ---")
    print(f"  samples : {n}")
    print(f"  min     : {int(x.min())}")
    print(f"  max     : {int(x.max())}")
    print(f"  peak |x|: {int(abs_x.max())}")
    print(f"  rms     : {float(np.sqrt(np.mean(x**2))):.2f}")
    print(f"  avg |x| : {float(np.mean(abs_x)):.2f}")
    h = histogram_bins(x)
    pct = {k: 100.0 * v / n for k, v in h.items()}
    print(f"  hist    : 0={h['0']:5d} ({pct['0']:5.1f}%)  "
          f"1-2={h['1-2']:5d} ({pct['1-2']:4.1f}%)  "
          f"3-10={h['3-10']:5d} ({pct['3-10']:4.1f}%)  "
          f"11-20={h['11-20']:5d} ({pct['11-20']:4.1f}%)  "
          f"21-40={h['21-40']:5d} ({pct['21-40']:4.1f}%)  "
          f"41+={h['41+']:5d} ({pct['41+']:4.1f}%)")


def make_plots(u5: np.ndarray, mx: np.ndarray,
               u5_label: str, max_label: str,
               save_prefix: str | None) -> None:
    t = np.arange(EXPECTED_LEN) / SAMPLE_RATE

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    # 1) waveforms
    axes[0].plot(t, u5, label=u5_label, color="tab:blue", linewidth=0.5)
    axes[0].plot(t, mx, label=max_label, color="tab:orange",
                 linewidth=0.5, alpha=0.7)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("int8 sample")
    axes[0].set_title("Waveform comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2) Power spectrum
    win = np.hanning(EXPECTED_LEN)
    u5_spec = np.abs(np.fft.rfft(u5 * win))
    mx_spec = np.abs(np.fft.rfft(mx * win))
    freqs = np.fft.rfftfreq(EXPECTED_LEN, 1.0 / SAMPLE_RATE)

    eps = 1e-9
    u5_db = 20.0 * np.log10(u5_spec + eps)
    mx_db = 20.0 * np.log10(mx_spec + eps)

    axes[1].plot(freqs, u5_db, label=u5_label, color="tab:blue")
    axes[1].plot(freqs, mx_db, label=max_label, color="tab:orange", alpha=0.7)
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("magnitude [dB]")
    axes[1].set_title("Power spectrum")
    axes[1].set_xlim(0, SAMPLE_RATE / 2)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3) Delta spectrum (smoothed)
    delta = u5_db - mx_db
    if HAVE_SCIPY:
        # boxcar smoothing for readability
        ksize = 64
        kernel = np.ones(ksize) / ksize
        delta_smooth = sp_signal.convolve(delta, kernel, mode="same")
    else:
        delta_smooth = delta

    axes[2].plot(freqs, delta, color="tab:gray", alpha=0.4, label="raw")
    axes[2].plot(freqs, delta_smooth, color="tab:red", linewidth=2,
                 label="smoothed")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("frequency [Hz]")
    axes[2].set_ylabel("U5 - MAX [dB]")
    axes[2].set_title("Delta spectrum (positive = U5 louder, negative = U5 attenuated)")
    axes[2].set_xlim(0, SAMPLE_RATE / 2)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 4) Spectrograms side by side (one row, two subplots really)
    # We use a single row with two stacked spectrograms via twin axes for clarity:
    #   easier: just plot both spectrograms in a 2x1 inset within axes[3].
    axes[3].axis("off")
    sub_u5 = fig.add_subplot(427)
    sub_mx = fig.add_subplot(428)

    if HAVE_SCIPY:
        f_u5, t_u5, S_u5 = sp_signal.spectrogram(
            u5, fs=SAMPLE_RATE, nperseg=256, noverlap=128, scaling="spectrum"
        )
        f_mx, t_mx, S_mx = sp_signal.spectrogram(
            mx, fs=SAMPLE_RATE, nperseg=256, noverlap=128, scaling="spectrum"
        )
        S_u5_db = 10.0 * np.log10(S_u5 + eps)
        S_mx_db = 10.0 * np.log10(S_mx + eps)
        vmax = max(S_u5_db.max(), S_mx_db.max())
        vmin = vmax - 60.0

        sub_u5.pcolormesh(t_u5, f_u5, S_u5_db, vmin=vmin, vmax=vmax, shading="auto")
        sub_u5.set_title(f"Spectrogram: {u5_label}")
        sub_u5.set_ylabel("Hz")
        sub_u5.set_xlabel("s")

        sub_mx.pcolormesh(t_mx, f_mx, S_mx_db, vmin=vmin, vmax=vmax, shading="auto")
        sub_mx.set_title(f"Spectrogram: {max_label}")
        sub_mx.set_ylabel("Hz")
        sub_mx.set_xlabel("s")
    else:
        sub_u5.text(0.5, 0.5, "install scipy for spectrograms",
                    ha="center", va="center")
        sub_mx.text(0.5, 0.5, "install scipy for spectrograms",
                    ha="center", va="center")

    fig.tight_layout()

    if save_prefix:
        out = f"{save_prefix}_compare.png"
        fig.savefig(out, dpi=120)
        print(f"\nSaved figure: {out}")

    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("u5_log", type=Path,
                        help="STM32U5 serial log containing AI_INPUT_DUMP block(s)")
    parser.add_argument("--list", action="store_true",
                        help="List all dumps in the U5 log with their predictions, then exit")
    parser.add_argument("--u5-dump-index", type=int, default=0,
                        help="Which dump block in the U5 log to use (default 0)")
    parser.add_argument("--max-log", type=Path, default=DEFAULT_MAX_LOG,
                        help=f"MAX78000 reference log (default: {DEFAULT_MAX_LOG})")
    parser.add_argument("--max-dump-index", type=int, default=0,
                        help="Which dump block in the MAX log to use (default 0)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plots as <prefix>_compare.png (default: show only)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't pop up a window (use with --save)")
    args = parser.parse_args()

    if args.no_show:
        os.environ["MPLBACKEND"] = "Agg"

    if not args.u5_log.exists():
        print(f"error: {args.u5_log} not found", file=sys.stderr)
        return 1
    if not args.max_log.exists():
        print(f"error: {args.max_log} not found", file=sys.stderr)
        return 1

    if args.list:
        text = args.u5_log.read_text(errors="replace")
        # Pair each dump with the prediction line that follows it
        # (form: "best index: N predicted: <name> logit_x1000: ...")
        pattern = re.compile(
            r"AI_INPUT_DUMP_BEGIN\s*\n(.*?)\nAI_INPUT_DUMP_END(.*?)"
            r"(best index:[^\r\n]+|$)",
            re.DOTALL,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            print(f"No AI_INPUT_DUMP blocks in {args.u5_log}")
            return 1
        print(f"\n{len(matches)} dump(s) in {args.u5_log}:")
        for i, m in enumerate(matches):
            pred = m.group(3).strip()
            print(f"  [{i}] {pred[:120]}")
        return 0

    # U5 dump is already signed int8 in the log
    u5_raw = extract_dump(args.u5_log, args.u5_dump_index, reinterpret_signed=False)
    if looks_uint8(u5_raw):
        print(f"note: U5 dump looks like uint8; reinterpreting as signed int8")
        u5 = np.where(u5_raw > 127, u5_raw - 256, u5_raw).astype(np.float32)
    else:
        u5 = u5_raw

    # MAX78000 reference dump is stored as uint8 bytes; reinterpret as signed
    mx_raw = extract_dump(args.max_log, args.max_dump_index, reinterpret_signed=False)
    if looks_uint8(mx_raw):
        mx = np.where(mx_raw > 127, mx_raw - 256, mx_raw).astype(np.float32)
    else:
        mx = mx_raw

    u5_label = f"U5: {args.u5_log.name}#{args.u5_dump_index}"
    max_label = f"MAX78000: {args.max_log.name}#{args.max_dump_index}"

    print(f"\n{u5_label}")
    print(f"{max_label}")
    print()
    print_summary(u5_label, u5)
    print()
    print_summary(max_label, mx)

    # Band energy comparison
    win = np.hanning(EXPECTED_LEN)
    u5_spec = np.abs(np.fft.rfft(u5 * win))
    mx_spec = np.abs(np.fft.rfft(mx * win))
    freqs = np.fft.rfftfreq(EXPECTED_LEN, 1.0 / SAMPLE_RATE)
    bands = [
        (0, 250),       # rumble / DC
        (250, 1000),    # voiced low (vowel fundamentals)
        (1000, 2000),   # voiced mid
        (2000, 4000),   # voiced high (formants)
        (4000, 6000),   # fricative low
        (6000, 8000),   # fricative high (/s/, /sh/, /f/)
    ]
    band_names = ["0-0.25", "0.25-1", "1-2", "2-4", "4-6", "6-8"]
    u5_db = band_energy_db(u5_spec, freqs, bands)
    mx_db = band_energy_db(mx_spec, freqs, bands)

    print("\n--- Band energy [dB] ---")
    print(f"  {'band [kHz]':<12} {'U5':>8} {'MAX':>8} {'U5-MAX':>8}")
    deltas = []
    for nm, u, m in zip(band_names, u5_db, mx_db):
        d = u - m
        deltas.append(d)
        print(f"  {nm:<12} {u:>8.1f} {m:>8.1f} {d:>8.1f}")

    # Dynamic hints based on the actual measured deltas
    print("\n--- Interpretation ---")
    avg_delta = float(np.mean(deltas))
    hf_delta = float(np.mean(deltas[-2:]))   # 4-6 + 6-8 kHz
    lf_delta = float(np.mean(deltas[:2]))    # 0-1 kHz
    voiced = float(np.mean(deltas[2:4]))     # 1-4 kHz

    if avg_delta > 6:
        print(f"  Broadband U5 energy is +{avg_delta:.1f} dB hotter than MAX78000.")
        print(f"  => U5 is louder overall.  Background floor is too high.")
    elif avg_delta < -6:
        print(f"  Broadband U5 energy is {avg_delta:.1f} dB quieter than MAX78000.")
        print(f"  => U5 is too quiet overall, model gets near-silence input.")
    else:
        print(f"  Broadband levels are within {abs(avg_delta):.1f} dB of MAX78000.")

    if hf_delta - lf_delta < -6:
        print(f"  HF (4-8 kHz) is {hf_delta - lf_delta:.1f} dB DOWN vs LF.")
        print(f"  => Classic SINC4 / MDF passband droop -- fricatives lose energy.")
    elif hf_delta - lf_delta > 6:
        print(f"  HF (4-8 kHz) is {hf_delta - lf_delta:+.1f} dB HOTTER than LF.")
        print(f"  => U5 has excess high-frequency content (mic noise / aliasing /")
        print(f"     MDF quantisation noise).  A low-pass would help, NOT a HF boost.")
    else:
        print(f"  HF/LF balance is roughly aligned ({hf_delta - lf_delta:+.1f} dB).")

    if abs(deltas[0]) > 10 and deltas[0] > deltas[1]:
        print(f"  0-0.25 kHz delta = {deltas[0]:+.1f} dB:")
        print(f"  => DC / rumble leakage on U5 -- HPF too soft.")


    if not args.no_show or args.save:
        make_plots(u5, mx, u5_label, max_label, args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
