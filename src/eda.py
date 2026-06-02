"""
eda.py — Initial Exploratory Data Analysis for UCI HAR
=======================================================
1. Class imbalance bar chart (6 activities)
2. Raw accelerometer signal plot for a single sample (Subject 1, WALKING)

Outputs saved to results/ as PNGs.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Add project root to path so we can import data_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_har_data, ACTIVITY_LABELS, CHANNEL_NAMES

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_class_distribution(y, subjects):
    """Bar chart of activity class counts with percentages."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count each activity
    labels_sorted = sorted(ACTIVITY_LABELS.keys())
    counts = [np.sum(y == label) for label in labels_sorted]
    names = [ACTIVITY_LABELS[label] for label in labels_sorted]
    total = len(y)

    # Color palette — warm-to-cool gradient
    colors = ["#FF6B6B", "#FFA07A", "#FFD93D", "#6BCB77", "#4D96FF", "#9B59B6"]

    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=1.5, zorder=3)

    # Add count + percentage labels on each bar
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("UCI HAR — Activity Class Distribution (All 10,299 Samples)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Activity", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Class distribution plot saved → {save_path}")
    plt.close()

    # Print summary table
    print("\n  Activity Class Counts:")
    print("  " + "-" * 45)
    for label in labels_sorted:
        count = np.sum(y == label)
        pct = 100.0 * count / total
        print(f"    {label}: {ACTIVITY_LABELS[label]:<22s}  {count:>5d}  ({pct:5.1f}%)")
    print("  " + "-" * 45)
    print(f"    {'TOTAL':<25s}  {total:>5d}")


def plot_raw_accelerometer(X, y, subjects):
    """
    Plot raw total accelerometer (X, Y, Z) for the first WALKING sample
    from Subject 1.
    """
    # Find first sample: subject=1, activity=1 (WALKING)
    mask = (subjects == 1) & (y == 1)
    idx = np.where(mask)[0][0]

    sample = X[idx]  # shape: (128, 9)
    timesteps = np.arange(128)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    acc_channels = [
        (0, "total_acc_x", "#FF6B6B"),
        (1, "total_acc_y", "#4D96FF"),
        (2, "total_acc_z", "#6BCB77"),
    ]

    for ax, (ch_idx, ch_name, color) in zip(axes, acc_channels):
        signal = sample[:, ch_idx]
        ax.plot(timesteps, signal, color=color, linewidth=1.5, alpha=0.9)
        ax.fill_between(timesteps, signal, alpha=0.15, color=color)
        ax.set_ylabel(ch_name, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 127)

    axes[0].set_title(
        f"Raw Total Accelerometer — Subject 1, WALKING (sample index={idx})",
        fontsize=13, fontweight="bold",
    )
    axes[-1].set_xlabel("Timestep (128 readings @ 50Hz = 2.56s window)", fontsize=11)

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "raw_accelerometer_subject1_walking.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Accelerometer signal plot saved → {save_path}")
    plt.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    data = load_har_data()

    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    print("\n" + "=" * 55)
    print("  INITIAL EDA")
    print("=" * 55)

    plot_class_distribution(y, subjects)
    plot_raw_accelerometer(X, y, subjects)

    print("\n  ✓ EDA complete. Check results/ for plots.\n")
