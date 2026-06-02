"""
rf_baseline.py — Random Forest Baseline with LOSO Cross-Validation
====================================================================
Trains a RandomForestClassifier on flattened raw inertial signals (1152 features)
using Leave-One-Subject-Out evaluation across all 30 subjects.
"""

import sys
import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_har_data, ACTIVITY_LABELS
from src.loso import get_loso_splits

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_rf_loso():
    """Run full 30-fold LOSO evaluation with Random Forest."""

    # ── Load data ──
    data = load_har_data()
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    print("\n" + "=" * 60)
    print("  RANDOM FOREST BASELINE — 30-FOLD LOSO")
    print("  Model: RandomForestClassifier(n_estimators=100)")
    print("  Features: Flattened raw signals (128 × 9 = 1152)")
    print("=" * 60)

    subject_accuracies = {}
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    for subject_id, X_train, X_test, y_train, y_test in get_loso_splits(X, subjects, y):
        fold_start = time.time()

        # Flatten 3D → 2D
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        # Train
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_flat, y_train)

        # Predict
        y_pred = clf.predict(X_test_flat)
        acc = accuracy_score(y_test, y_pred)
        n_correct = np.sum(y_pred == y_test)

        subject_accuracies[subject_id] = acc
        total_correct += n_correct
        total_samples += len(y_test)

        fold_time = time.time() - fold_start
        print(f"  Fold {subject_id:2d} | Subject {subject_id:2d} | "
              f"Test samples: {len(y_test):4d} | "
              f"Acc: {acc:.4f} ({n_correct}/{len(y_test)}) | "
              f"Time: {fold_time:.1f}s")

    elapsed = time.time() - start_time

    # ── Summary ──
    mean_acc = np.mean(list(subject_accuracies.values()))
    std_acc = np.std(list(subject_accuracies.values()))
    overall_acc = total_correct / total_samples

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Mean subject accuracy:    {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Overall accuracy:         {overall_acc:.4f} ({total_correct}/{total_samples})")
    print(f"  Best  subject:            {max(subject_accuracies, key=subject_accuracies.get)} "
          f"({max(subject_accuracies.values()):.4f})")
    print(f"  Worst subject:            {min(subject_accuracies, key=subject_accuracies.get)} "
          f"({min(subject_accuracies.values()):.4f})")
    print(f"  Total time:               {elapsed:.1f}s")
    print("=" * 60)

    # ── Save to file ──
    save_path = os.path.join(RESULTS_DIR, "rf_loso_results.txt")
    with open(save_path, "w") as f:
        f.write("Random Forest Baseline — LOSO Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: RandomForestClassifier(n_estimators=100)\n")
        f.write(f"Features: Flattened raw signals (1152)\n\n")
        f.write(f"{'Subject':>8s}  {'Samples':>8s}  {'Accuracy':>10s}\n")
        f.write("-" * 30 + "\n")
        for sid in sorted(subject_accuracies.keys()):
            n_samples = np.sum(subjects == sid)
            f.write(f"{sid:>8d}  {n_samples:>8d}  {subject_accuracies[sid]:>10.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(f"\nMean accuracy:    {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")

    print(f"\n  ✓ Results saved → {save_path}\n")

    return subject_accuracies, mean_acc


if __name__ == "__main__":
    run_rf_loso()
