"""
plot_domain_shift.py — Visualize Per-Subject LOSO Accuracy (1D CNN)
====================================================================
Highlights "problem subjects" caused by inter-subject domain shift.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1D CNN per-subject LOSO accuracies (from cnn_loso.py run) ──
CNN_ACCURACIES = {
    1:  1.0000,  2:  0.9073,  3:  1.0000,  4:  0.9558,  5:  0.8444,
    6:  0.9662,  7:  0.9578,  8:  0.9502,  9:  0.8507,  10: 0.7075,
    11: 1.0000,  12: 0.9906,  13: 0.9755,  14: 0.6935,  15: 1.0000,
    16: 0.7732,  17: 0.9538,  18: 0.9945,  19: 0.9972,  20: 1.0000,
    21: 0.9828,  22: 1.0000,  23: 0.9462,  24: 1.0000,  25: 0.8460,
    26: 0.9974,  27: 1.0000,  28: 0.8455,  29: 0.9157,  30: 0.9373,
}

PROBLEM_SUBJECTS = {9, 10, 14, 16}
MEAN_ACC = np.mean(list(CNN_ACCURACIES.values()))

# ── Plot ──
subjects = sorted(CNN_ACCURACIES.keys())
accs = [CNN_ACCURACIES[s] for s in subjects]
colors = ["#E74C3C" if s in PROBLEM_SUBJECTS else "#2ECC71" for s in subjects]

fig, ax = plt.subplots(figsize=(10, 10))

bars = ax.barh(
    [f"Subject {s}" for s in subjects],
    [a * 100 for a in accs],
    color=colors,
    edgecolor="white",
    linewidth=0.8,
    height=0.7,
    zorder=3,
)

# Add accuracy labels on each bar
for bar, acc in zip(bars, accs):
    x_pos = bar.get_width() + 0.5
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{acc * 100:.1f}%", va="center", fontsize=8.5, fontweight="bold")

# Mean accuracy line
ax.axvline(MEAN_ACC * 100, color="#3498DB", linestyle="--", linewidth=2, zorder=4, label=f"Mean: {MEAN_ACC*100:.1f}%")

# Danger zone shading (below 80%)
ax.axvspan(0, 80, alpha=0.08, color="red", zorder=0)
ax.axvline(80, color="#E74C3C", linestyle=":", linewidth=1, alpha=0.5, zorder=1)

# Legend
good_patch = mpatches.Patch(color="#2ECC71", label="Normal subjects")
bad_patch = mpatches.Patch(color="#E74C3C", label="Problem subjects (domain shift)")
mean_line = plt.Line2D([0], [0], color="#3498DB", linestyle="--", linewidth=2, label=f"Mean accuracy: {MEAN_ACC*100:.1f}%")
ax.legend(handles=[good_patch, bad_patch, mean_line], loc="lower right", fontsize=10)

ax.set_xlabel("LOSO Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title("1D CNN Per-Subject LOSO Accuracy -- Domain Shift Analysis",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlim(0, 108)
ax.grid(axis="x", alpha=0.3, zorder=0)
ax.invert_yaxis()

plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, "per_subject_accuracy.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\n  Plot saved -> {save_path}")
plt.close()

# ── Print summary ──
print("\n  Problem subjects (accuracy < 80%):")
for s in sorted(PROBLEM_SUBJECTS):
    print(f"    Subject {s:2d}: {CNN_ACCURACIES[s]*100:.1f}%")
print(f"\n  Mean accuracy (all 30): {MEAN_ACC*100:.1f}%")
print(f"  Mean accuracy (excluding problem 4): "
      f"{np.mean([a for s, a in CNN_ACCURACIES.items() if s not in PROBLEM_SUBJECTS])*100:.1f}%")
