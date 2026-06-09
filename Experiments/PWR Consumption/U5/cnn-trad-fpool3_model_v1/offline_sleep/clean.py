#!/usr/bin/env python3
"""Trim a raw PPK2 trace from the first current threshold crossing.

The script scans a CSV with `Timestamp(ms),Current(uA)`, finds the first row
whose current is at or above the threshold, re-zeros time at that point, and
keeps only the following time window.
"""

import argparse
import csv
from pathlib import Path


def find_anchor(rows, threshold_ua):
    for line_number, row in rows:
        if len(row) < 2:
            continue
        try:
            timestamp_ms = float(row[0])
            current_ua = float(row[1])
        except ValueError:
            continue
        if current_ua >= threshold_ua:
            return line_number, timestamp_ms
    return None, None


def clean_trace(input_path, output_path, threshold_ma, window_ms):
    threshold_ua = threshold_ma * 1000.0

    with input_path.open(newline="") as src:
        reader = csv.reader(src)
        header = next(reader, None)
        anchor_line, anchor_time_ms = find_anchor(enumerate(reader, start=2), threshold_ua)
    if anchor_time_ms is None:
        raise RuntimeError(
            f"No sample reached {threshold_ma:.3f} mA in {input_path}"
        )

    selected_rows = []
    with input_path.open(newline="") as src:
        reader = csv.reader(src)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                timestamp_ms = float(row[0])
                current_ua = float(row[1])
            except ValueError:
                continue

            relative_time_ms = timestamp_ms - anchor_time_ms
            if relative_time_ms < 0:
                continue
            if relative_time_ms > window_ms:
                break

            selected_rows.append((relative_time_ms, current_ua))

    with output_path.open("w", newline="") as dst:
        writer = csv.writer(dst)
        if header is not None:
            writer.writerow(header)
        for relative_time_ms, current_ua in selected_rows:
            writer.writerow([
                f"{relative_time_ms:.12g}",
                f"{current_ua:.12g}",
            ])

    return anchor_line, anchor_time_ms, len(selected_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Trim a raw PPK2 CSV from the first threshold crossing."
    )
    parser.add_argument(
        "--input",
        default=Path(__file__).with_name("on_raw.csv"),
        type=Path,
        help="raw CSV to clean",
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).with_name("on_clean.csv"),
        type=Path,
        help="output cleaned CSV",
    )
    parser.add_argument(
        "--threshold-ma",
        type=float,
        default=10.0,
        help="first current threshold in mA",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=30000.0,
        help="time window to keep after the threshold crossing, in ms",
    )
    args = parser.parse_args()

    anchor_line, anchor_time_ms, kept_rows = clean_trace(
        args.input,
        args.output,
        args.threshold_ma,
        args.window_ms,
    )

    print(
        f"anchor line {anchor_line}, time {anchor_time_ms:.6f} ms, "
        f"kept {kept_rows} rows -> {args.output}"
    )


if __name__ == "__main__":
    main()