#!/usr/bin/env python3
"""Plot power traces for multiple platforms in overlapping subplots."""

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

    # Load all traces
    coral_path = base_dir / "Coral/cnn-trad-fpool3_model_v21/offline/on_clean.csv"
    gap9_path = base_dir / "Gap9/cnn-trad-fpool3_model_v2/offline/on_clean.csv"
    max_path = base_dir / "MAX78000/cnn-trad-fpool3_model_v1/offline_sleep/on_clean.csv"
    stm_path = base_dir / "U5/cnn-trad-fpool3_model_v1/offline_sleep/on_clean.csv"

    print("Loading traces...")
    coral_time, coral_curr = read_csv(coral_path)
    gap9_time, gap9_curr = read_csv(gap9_path)
    max_time, max_curr = read_csv(max_path)
    stm_time, stm_curr = read_csv(stm_path)

    print(f"Coral: {len(coral_curr)} pts, mean {coral_curr.mean():.2f} mA")
    print(f"GAP9:  {len(gap9_curr)} pts, mean {gap9_curr.mean():.2f} mA")
    print(f"MAX:   {len(max_curr)} pts, mean {max_curr.mean():.2f} mA")
    print(f"STM:   {len(stm_curr)} pts, mean {stm_curr.mean():.2f} mA")

    # Create 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot 1: GAP9 only
    axes[0].plot(gap9_time, gap9_curr, color='green', linewidth=0.5, label='GAP9')
    axes[0].set_title('GAP9', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Current (mA)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: GAP9 + STM (STM first so it doesn't eclipse GAP9)
    axes[1].plot(stm_time, stm_curr, color='blue', linewidth=0.5, label='STM')
    axes[1].plot(gap9_time, gap9_curr, color='green', linewidth=0.5, label='GAP9')
    axes[1].set_title('GAP9 + STM', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Current (mA)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: GAP9 + STM + MAX (larger values first)
    axes[2].plot(max_time, max_curr, color='orange', linewidth=0.5, label='MAX')
    axes[2].plot(stm_time, stm_curr, color='blue', linewidth=0.5, label='STM')
    axes[2].plot(gap9_time, gap9_curr, color='green', linewidth=0.5, label='GAP9')
    axes[2].set_title('GAP9 + STM + MAX', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Current (mA)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: GAP9 + STM + MAX + Coral (Coral first, smallest on top)
    axes[3].plot(coral_time, coral_curr, color='red', linewidth=0.5, label='Coral')
    axes[3].plot(max_time, max_curr, color='orange', linewidth=0.5, label='MAX')
    axes[3].plot(stm_time, stm_curr, color='blue', linewidth=0.5, label='STM')
    axes[3].plot(gap9_time, gap9_curr, color='green', linewidth=0.5, label='GAP9')
    axes[3].set_title('GAP9 + STM + MAX + Coral', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Time (ms)')
    axes[3].set_ylabel('Current (mA)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / "platforms_overlay.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
