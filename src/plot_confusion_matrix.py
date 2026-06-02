"""
plot_confusion_matrix.py — Normalized Confusion Matrix (1D CNN + Z-score)
==========================================================================
Loads saved predictions from error_analysis.py and plots a publication-ready
confusion matrix showing inter-activity classification errors.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Activity labels (1-indexed to match y_true/y_pred values)
ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING\nUPSTAIRS",
    3: "WALKING\nDOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# Short names for classification report
ACTIVITY_SHORT = [
    "WALKING", "WALK_UP", "WALK_DOWN", "SITTING", "STANDING", "LAYING"
]


def plot_confusion_matrix():
    """Load saved predictions and plot normalized confusion matrix."""

    # ── Load predictions ──
    y_true = np.load(os.path.join(RESULTS_DIR, "y_true_all.npy"))
    y_pred = np.load(os.path.join(RESULTS_DIR, "y_pred_all.npy"))
    print(f"\n  Loaded predictions: {len(y_true)} samples")
    print(f"  Overall accuracy: {np.mean(y_true == y_pred)*100:.2f}%")

    # ── Compute confusion matrix ──
    labels = [1, 2, 3, 4, 5, 6]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ── Print raw confusion matrix ──
    print("\n  Raw confusion matrix:")
    print(f"  {'':>12s}", end="")
    for name in ACTIVITY_SHORT:
        print(f"  {name:>9s}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {ACTIVITY_SHORT[i]:>12s}", end="")
        for val in row:
            print(f"  {val:>9d}", end="")
        print()

    # ── Print classification report ──
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels,
                                 target_names=ACTIVITY_SHORT, digits=4))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Annotation: show both percentage and count
    annot = np.empty_like(cm_normalized, dtype=object)
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            pct = cm_normalized[i, j] * 100
            count = cm[i, j]
            if pct >= 1.0:
                annot[i, j] = f"{pct:.1f}%\n({count})"
            elif count > 0:
                annot[i, j] = f"{pct:.1f}%\n({count})"
            else:
                annot[i, j] = ""

    display_names = [ACTIVITY_NAMES[i] for i in labels]

    sns.heatmap(
        cm_normalized * 100,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        vmin=0, vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Classification Rate (%)"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Activity", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Activity", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Normalized 1D CNN -- LOSO Confusion Matrix\n"
                 "(Subject-wise Z-score Normalization | Accuracy: 95.8%)",
                 fontsize=13, fontweight="bold", pad=15)

    # Rotate tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved -> {save_path}")
    plt.close()

    # ── Highlight key error pairs ──
    print("\n  Top confused activity pairs (off-diagonal > 1%):")
    for i in range(6):
        for j in range(6):
            if i != j and cm_normalized[i, j] >= 0.01:
                pct = cm_normalized[i, j] * 100
                print(f"    {ACTIVITY_SHORT[i]:>12s} -> {ACTIVITY_SHORT[j]:<12s}  "
                      f"{pct:.1f}% ({cm[i, j]} samples)")


if __name__ == "__main__":
    plot_confusion_matrix()
