#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

KWS20 = [
    "up", "down", "left", "right", "stop", "go",
    "yes", "no", "on", "off",
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)
    data_root = Path(args.data_root).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label in KWS20:
        wavs = sorted((data_root / label).glob("*.wav"))
        chosen = random.sample(wavs, min(args.n_per_class, len(wavs)))
        for wav in chosen:
            rows.append({"label": label, "wav": str(wav.resolve())})

    random.shuffle(rows)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "wav"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} trials to {out}")

if __name__ == "__main__":
    main()
