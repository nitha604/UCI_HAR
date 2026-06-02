"""
loso.py — Leave-One-Subject-Out (LOSO) Cross-Validation Utilities
===================================================================
Provides a generator for LOSO splits and evaluation helpers
for the UCI HAR cross-subject recognition project.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_har_data


# ──────────────────────────────────────────────
# LOSO Generator
# ──────────────────────────────────────────────
def get_loso_splits(X, subjects, y):
    """
    Generator that yields one Leave-One-Subject-Out fold per iteration.

    For each unique subject ID (sorted 1→30):
      - TEST  = all samples belonging to that subject
      - TRAIN = everything else

    Yields:
        (subject_id, X_train, X_test, y_train, y_test)
    """
    for subject_id in sorted(np.unique(subjects)):
        test_mask = (subjects == subject_id)
        train_mask = ~test_mask
        yield (
            subject_id,
            X[train_mask], X[test_mask],
            y[train_mask], y[test_mask],
        )


# ──────────────────────────────────────────────
# Quick test — first fold only
# ──────────────────────────────────────────────
if __name__ == "__main__":
    data = load_har_data()
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    print("\n" + "=" * 55)
    print("  LOSO GENERATOR TEST — FOLD 1 ONLY")
    print("=" * 55)

    # Grab just the first fold
    gen = get_loso_splits(X, subjects, y)
    subject_id, X_train, X_test, y_train, y_test = next(gen)

    print(f"  Held-out subject:  {subject_id}")
    print(f"  X_train shape:     {X_train.shape}")
    print(f"  X_test  shape:     {X_test.shape}")
    print(f"  y_train shape:     {y_train.shape}")
    print(f"  y_test  shape:     {y_test.shape}")
    print(f"  Train samples:     {X_train.shape[0]}")
    print(f"  Test  samples:     {X_test.shape[0]}")
    print(f"  Total:             {X_train.shape[0] + X_test.shape[0]}  (should be 10299)")
    print(f"  Train activities:  {sorted(np.unique(y_train).tolist())}")
    print(f"  Test  activities:  {sorted(np.unique(y_test).tolist())}")
    print("=" * 55)

    # ── STEP 2: Flatten test ──
    print("\n" + "=" * 55)
    print("  FLATTEN TEST — FOLD 1 (for Random Forest)")
    print("=" * 55)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    print(f"  X_train 3D:       {X_train.shape}  →  X_train_flat: {X_train_flat.shape}")
    print(f"  X_test  3D:       {X_test.shape}   →  X_test_flat:  {X_test_flat.shape}")
    print(f"  Features per sample: 128 × 9 = {128 * 9}")
    print("=" * 55)
