#!/usr/bin/env python3
"""Plot GAP9 power trace with zoom detail."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def read_csv(csv_path):
    """Read CSV and return times (ms) and currents (mA)."""
    times = []
    currents = []

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                time_ms = float(row[0])
                current_ua = float(row[1])
                times.append(time_ms)
                currents.append(current_ua / 1000.0)  # Convert to mA
            except ValueError:
                continue

    return np.array(times), np.array(currents)


def main():
    base_dir = Path(__file__).parent.parent

    # Load GAP9 trace
    gap9_path = base_dir / "Gap9/cnn-trad-fpool3_model_v2/offline/on_clean.csv"

    print("Loading GAP9 trace...")
    gap9_time, gap9_curr = read_csv(gap9_path)

    print(f"GAP9: {len(gap9_curr)} pts, mean {gap9_curr.mean():.2f} mA")

    # Create 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Full trace
    axes[0].plot(gap9_time, gap9_curr, color='green', linewidth=0.5)
    axes[0].set_title('GAP9 - Full Trace', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Current (mA)')
    axes[0].grid(True, alpha=0.3)
    # Add a shaded region to show the zoom area
    axes[0].axvspan(5270, 5278, alpha=0.2, color='yellow')

    # Plot 2: Zoomed to 5270-5278 ms
    zoom_mask = (gap9_time >= 5270) & (gap9_time <= 5278)
    zoom_time = gap9_time[zoom_mask]
    zoom_curr = gap9_curr[zoom_mask]

    axes[1].plot(zoom_time, zoom_curr, color='green', linewidth=0.8)
    axes[1].set_title('GAP9 - Zoomed (5270-5278 ms)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Current (mA)')
    axes[1].set_xlim(5270, 5278)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / "gap9_zoom.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
