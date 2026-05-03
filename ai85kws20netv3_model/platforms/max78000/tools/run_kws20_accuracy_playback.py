#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import serial

KEYWORDS = [
    "up", "down", "left", "right", "stop", "go",
    "yes", "no", "on", "off",
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
]

def load_manifest(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().lower()
            wav = Path(row["wav"]).expanduser()
            if label not in KEYWORDS and label not in ["unknown", "silence"]:
                raise ValueError(f"Unknown label in manifest: {label}")
            if label != "silence" and not wav.exists():
                raise FileNotFoundError(wav)
            rows.append({"label": label, "wav": str(wav)})
    return rows

def parse_prediction(line):
    s = line.strip().lower()

    if "cnn time" in s or "bench" in s:
        return None

    # KWS demo usually prints result/confidence style lines.
    looks_like_result = any(
        token in s for token in [
            "detected", "keyword", "class", "classification",
            "result", "word", "confidence", "prob"
        ]
    )

    if not looks_like_result:
        return None

    found = []
    for kw in KEYWORDS + ["unknown"]:
        if re.search(rf"\b{re.escape(kw)}\b", s):
            found.append(kw)

    if len(found) == 1:
        return found[0]

    return None

def read_for_prediction(ser, response_window_s, raw_log):
    start = time.time()
    predictions = []

    while time.time() - start < response_window_s:
        data = ser.readline()
        if not data:
            continue

        ts = datetime.now().isoformat(timespec="milliseconds")
        line = data.decode("utf-8", errors="replace").strip()

        raw_log.write(f"{ts},{line}\n")
        raw_log.flush()
        print(line)

        pred = parse_prediction(line)
        if pred is not None:
            predictions.append(pred)

    if predictions:
        return predictions[-1]

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out-root", default=str(Path.home() / "max78000/measurements"))
    parser.add_argument("--response-window", type=float, default=2.5)
    parser.add_argument("--pre-delay", type=float, default=0.7)
    parser.add_argument("--post-delay", type=float, default=0.8)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--player", default="aplay")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    trials = manifest * args.repeat
    if args.shuffle:
        random.shuffle(trials)

    out_dir = Path(args.out_root) / datetime.now().strftime("%Y%m%d_%H%M%S_kws20_accuracy")
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_csv = out_dir / "accuracy_trials.csv"
    raw_log_path = out_dir / "serial_raw_accuracy.log"
    summary_path = out_dir / "accuracy_summary.json"

    print(f"Output: {out_dir}")
    print(f"Opening serial: {args.port} @ {args.baud}")

    results = []

    with serial.Serial(args.port, args.baud, timeout=0.2) as ser, \
            open(raw_log_path, "w") as raw_log, \
            open(trial_csv, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial", "expected", "predicted", "correct",
                "wav", "start_time_iso", "end_time_iso"
            ],
        )
        writer.writeheader()

        time.sleep(1.0)
        ser.reset_input_buffer()

        for i, trial in enumerate(trials, start=1):
            expected = trial["label"]
            wav = trial["wav"]
            start_iso = datetime.now().isoformat(timespec="milliseconds")

            print("\n========================================")
            print(f"Trial {i}/{len(trials)}")
            print(f"Expected: {expected}")
            print(f"WAV:      {wav}")

            time.sleep(args.pre_delay)

            if expected == "silence":
                print("Silence trial, no playback")
            else:
                subprocess.run([args.player, wav], check=True)

            predicted = read_for_prediction(
                ser=ser,
                response_window_s=args.response_window,
                raw_log=raw_log,
            )

            time.sleep(args.post_delay)

            correct = int(predicted == expected)
            end_iso = datetime.now().isoformat(timespec="milliseconds")

            row = {
                "trial": i,
                "expected": expected,
                "predicted": predicted if predicted is not None else "",
                "correct": correct,
                "wav": wav,
                "start_time_iso": start_iso,
                "end_time_iso": end_iso,
            }

            writer.writerow(row)
            f.flush()
            results.append(row)

            print(f"Predicted: {row['predicted']}")
            print(f"Correct:   {bool(correct)}")

    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total else 0.0

    per_label = {}
    for r in results:
        label = r["expected"]
        per_label.setdefault(label, {"total": 0, "correct": 0})
        per_label[label]["total"] += 1
        per_label[label]["correct"] += r["correct"]

    for label, d in per_label.items():
        d["accuracy"] = d["correct"] / d["total"] if d["total"] else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "manifest": str(Path(args.manifest).resolve()),
        "port": args.port,
        "baud": args.baud,
        "total_trials": total,
        "correct_trials": correct,
        "accuracy": accuracy,
        "per_label": per_label,
        "output_dir": str(out_dir),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========================================")
    print("DONE")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Output:   {out_dir}")

if __name__ == "__main__":
    main()
