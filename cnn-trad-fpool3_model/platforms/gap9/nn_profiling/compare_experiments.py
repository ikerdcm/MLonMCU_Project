#!/usr/bin/env python3
"""
compare_experiments.py — Compare performance across multiple profiling experiments.

Usage:
    # Compare all experiments matching a glob
    python compare_experiments.py "experiments/mobilenet_v1_0_25_128_quant_ne16_*"

    # Compare specific experiments
    python compare_experiments.py experiments/exp_a experiments/exp_b

    # Save to CSV
    python compare_experiments.py "experiments/*" --csv comparison.csv
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def load_experiment(exp_dir: str) -> dict | None:
    """Load results.json from an experiment directory."""
    results_path = os.path.join(exp_dir, "results.json")
    if not os.path.isfile(results_path):
        return None
    with open(results_path) as f:
        return json.load(f)


def get_experiment_label(exp_dir: str) -> str:
    """Derive a short label for an experiment from its directory name or cli_args."""
    cli_path = os.path.join(exp_dir, "cli_args.yml")
    if os.path.isfile(cli_path):
        try:
            import yaml
            with open(cli_path) as f:
                cli = yaml.safe_load(f)
            parts = []
            if cli.get("scheme"):
                parts.append(cli["scheme"])
            if cli.get("platform"):
                parts.append(cli["platform"])
            # Append timestamp from dir name (last two _-separated segments)
            name = Path(exp_dir).name
            tokens = name.rsplit("_", 2)
            if len(tokens) >= 3:
                parts.append(f"{tokens[-2]}_{tokens[-1]}")
            if parts:
                return "_".join(parts)
        except Exception:
            pass
    return Path(exp_dir).name


def parse_performance(data: dict) -> dict:
    """
    Extract per-layer performance from results data.
    Returns {layer_name: {"cycles": int, "ops": int, "ops_cyc": float}}
    """
    layers = {}
    for row in data.get("performance", []):
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        name, cycles, ops, ops_cyc = row[0], row[1], row[2], row[3]
        if name == "IO_Wait":
            continue
        layers[name] = {
            "cycles": cycles,
            "ops": ops,
            "ops_cyc": ops_cyc,
        }
    return layers


def build_comparison_table(experiments: list[tuple[str, dict]]) -> tuple[list[str], list[str], list[list]]:
    """
    Build a comparison table from loaded experiments.

    Returns:
        headers: column headers
        layer_names: ordered list of layer names
        rows: list of rows, each row is [layer_name, ops, cyc_1, ops_cyc_1, cyc_2, ops_cyc_2, ...]
    """
    labels = [label for label, _ in experiments]
    all_perf = [parse_performance(data) for _, data in experiments]

    # Collect all layer names in order of first appearance
    seen = set()
    layer_order = []
    for perf in all_perf:
        for name in perf:
            if name not in seen:
                seen.add(name)
                layer_order.append(name)

    # Build headers
    headers = ["Layer", "Ops"]
    for label in labels:
        headers.append(f"Cycles ({label})")
        headers.append(f"Ops/Cyc ({label})")

    # Build rows
    rows = []
    for layer in layer_order:
        # Use ops from the first experiment that has this layer
        ops = None
        for perf in all_perf:
            if layer in perf:
                ops = perf[layer]["ops"]
                break

        row = [layer, ops if ops is not None else "—"]
        for perf in all_perf:
            if layer in perf:
                row.append(perf[layer]["cycles"])
                oc = perf[layer]["ops_cyc"]
                row.append(f"{oc:.2f}" if oc == oc else "—")  # NaN check
            else:
                row.append("—")
                row.append("—")
        rows.append(row)

    return headers, layer_order, rows


def print_table(headers: list[str], rows: list[list], use_color: bool = True):
    """Print a formatted comparison table to stdout."""
    BOLD = "\033[1m" if use_color else ""
    CYAN = "\033[36m" if use_color else ""
    GREEN = "\033[32m" if use_color else ""
    YELLOW = "\033[33m" if use_color else ""
    DIM = "\033[2m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    # Calculate column widths
    str_rows = [[str(c) for c in row] for row in rows]
    col_widths = [max(len(h), *(len(r[i]) for r in str_rows)) for i, h in enumerate(headers)]

    # Print header
    header_line = "  ".join(f"{BOLD}{h:<{w}}{RESET}" for h, w in zip(headers, col_widths))
    sep_line = f"{DIM}{'─' * (sum(col_widths) + 2 * (len(col_widths) - 1))}{RESET}"

    print(sep_line)
    print(header_line)
    print(sep_line)

    # Print rows
    for str_row in str_rows:
        name = str_row[0]
        if name == "Total":
            print(sep_line)
            parts = [f"{BOLD}{YELLOW}{str_row[0]:<{col_widths[0]}}{RESET}"]
            for i in range(1, len(str_row)):
                parts.append(f"{BOLD}{str_row[i]:<{col_widths[i]}}{RESET}")
            print("  ".join(parts))
        else:
            parts = [f"{GREEN}{str_row[0]:<{col_widths[0]}}{RESET}"]
            parts.append(f"{str_row[1]:<{col_widths[1]}}")
            for i in range(2, len(str_row)):
                parts.append(f"{str_row[i]:<{col_widths[i]}}")
            print("  ".join(parts))

    print(sep_line)


def save_csv(headers: list[str], rows: list[list], path: str):
    """Save comparison table to CSV."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Saved CSV to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance across profiling experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment directories or glob patterns (e.g. 'experiments/mobilenet_*')",
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        help="Save comparison table to a CSV file",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    args = parser.parse_args()

    # Resolve globs
    exp_dirs = []
    for pattern in args.experiments:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f"Warning: no matches for '{pattern}'", file=sys.stderr)
        exp_dirs.extend(matches)

    # Load experiments
    experiments = []
    for d in exp_dirs:
        if not os.path.isdir(d):
            continue
        data = load_experiment(d)
        if data is None:
            print(f"Skipping {d}: no results.json", file=sys.stderr)
            continue
        label = get_experiment_label(d)
        experiments.append((label, data))

    if len(experiments) < 1:
        print("No valid experiments found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nComparing {len(experiments)} experiment(s):\n")
    for label, _ in experiments:
        print(f"  • {label}")
    print()

    headers, _, rows = build_comparison_table(experiments)
    print_table(headers, rows, use_color=not args.no_color)

    if args.csv:
        save_csv(headers, rows, args.csv)


if __name__ == "__main__":
    main()
