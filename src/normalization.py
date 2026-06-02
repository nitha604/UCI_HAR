"""
normalization.py — Subject-Wise Z-Score Normalization
=======================================================
Removes inter-subject domain shift by normalizing each subject's
sensor data to zero-mean, unit-variance independently per channel.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_har_data, CHANNEL_NAMES


def normalize_subject_wise(X, subjects, eps=1e-8):
    """
    Apply per-subject, per-channel Z-score normalization.

    For each subject s and each channel c:
        X_norm[subject_s, :, c] = (X[subject_s, :, c] - mean_sc) / (std_sc + eps)

    where mean_sc and std_sc are computed over ALL timesteps and ALL samples
    belonging to subject s for channel c.

    Args:
        X: np.ndarray of shape (N, 128, 9)
        subjects: np.ndarray of shape (N,) — subject IDs
        eps: small constant to prevent division by zero

    Returns:
        X_norm: np.ndarray of shape (N, 128, 9), normalized copy
    """
    X_norm = X.copy().astype(np.float64)
    unique_subjects = sorted(np.unique(subjects))

    for subject_id in unique_subjects:
        mask = (subjects == subject_id)
        # X[mask] has shape (N_s, 128, 9)
        subject_data = X_norm[mask]

        for ch in range(X.shape[2]):
            # Compute mean/std across ALL timesteps and ALL samples for this subject+channel
            ch_data = subject_data[:, :, ch]  # shape: (N_s, 128)
            mean = ch_data.mean()
            std = ch_data.std()
            X_norm[mask, :, ch] = (ch_data - mean) / (std + eps)

    return X_norm


# ──────────────────────────────────────────────
# Test: before vs after normalization
# ──────────────────────────────────────────────
if __name__ == "__main__":
    data = load_har_data()
    X = data["X"]
    subjects = data["subjects"]

    print("\n" + "=" * 65)
    print("  SUBJECT-WISE Z-SCORE NORMALIZATION TEST")
    print("=" * 65)

    # ── Before normalization: Subject 1 stats ──
    mask_s1 = (subjects == 1)
    X_s1 = X[mask_s1]  # shape: (N_s1, 128, 9)
    print(f"\n  Subject 1: {mask_s1.sum()} samples")
    print(f"\n  BEFORE normalization (Subject 1):")
    print(f"  {'Channel':<15s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}")
    print("  " + "-" * 58)
    for ch in range(9):
        ch_data = X_s1[:, :, ch]
        print(f"  {CHANNEL_NAMES[ch]:<15s}  {ch_data.mean():>10.4f}  {ch_data.std():>10.4f}  "
              f"{ch_data.min():>10.4f}  {ch_data.max():>10.4f}")

    # ── Normalize ──
    print("\n  Normalizing all 10,299 samples...")
    X_norm = normalize_subject_wise(X, subjects)
    print(f"  Done! Shape: {X_norm.shape}")

    # ── After normalization: Subject 1 stats ──
    X_s1_norm = X_norm[mask_s1]
    print(f"\n  AFTER normalization (Subject 1):")
    print(f"  {'Channel':<15s}  {'Mean':>10s}  {'Std':>10s}  {'Min':>10s}  {'Max':>10s}")
    print("  " + "-" * 58)
    for ch in range(9):
        ch_data = X_s1_norm[:, :, ch]
        print(f"  {CHANNEL_NAMES[ch]:<15s}  {ch_data.mean():>10.4f}  {ch_data.std():>10.4f}  "
              f"{ch_data.min():>10.4f}  {ch_data.max():>10.4f}")

    # ── Verify a different subject too ──
    mask_s10 = (subjects == 10)
    X_s10_norm = X_norm[mask_s10]
    print(f"\n  AFTER normalization (Subject 10 -- problem subject):")
    print(f"  {'Channel':<15s}  {'Mean':>10s}  {'Std':>10s}")
    print("  " + "-" * 38)
    for ch in range(9):
        ch_data = X_s10_norm[:, :, ch]
        print(f"  {CHANNEL_NAMES[ch]:<15s}  {ch_data.mean():>10.6f}  {ch_data.std():>10.6f}")

    print("\n" + "=" * 65)
    print("  Expected: mean ~ 0.0, std ~ 1.0 for all channels")
    print("=" * 65)
