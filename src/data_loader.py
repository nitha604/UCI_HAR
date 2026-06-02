"""
data_loader.py — UCI HAR Raw Inertial Signal Loader
=====================================================
Loads the RAW 9-channel sensor data from the Inertial Signals folder,
NOT the pre-computed 561-feature vectors (X_train.txt / X_test.txt).

Output shapes:
    X:        (10299, 128, 9)   — Samples × Timesteps × Channels
    y:        (10299,)          — Activity labels (1-6)
    subjects: (10299,)          — Subject IDs (1-30)

Channel order:
    0: total_acc_x    3: body_acc_x    6: body_gyro_x
    1: total_acc_y    4: body_acc_y    7: body_gyro_y
    2: total_acc_z    5: body_acc_z    8: body_gyro_z
"""

import os
import numpy as np

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data_uci_har",
    "UCI HAR Dataset",
)

# The 9 raw signal files, in a fixed canonical order
SIGNAL_FILES = [
    "total_acc_x_{}.txt",
    "total_acc_y_{}.txt",
    "total_acc_z_{}.txt",
    "body_acc_x_{}.txt",
    "body_acc_y_{}.txt",
    "body_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
]

CHANNEL_NAMES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


# ──────────────────────────────────────────────
# Core loading functions
# ──────────────────────────────────────────────
def _load_signals(split: str) -> np.ndarray:
    """
    Load all 9 inertial signal files for a given split ('train' or 'test').

    Returns:
        np.ndarray of shape (N_samples, 128, 9)
    """
    signal_dir = os.path.join(BASE_DIR, split, "Inertial Signals")
    channels = []
    for sig_template in SIGNAL_FILES:
        fpath = os.path.join(signal_dir, sig_template.format(split))
        # Each file: rows = samples, cols = 128 timesteps (whitespace-delimited)
        data = np.loadtxt(fpath)
        channels.append(data)
    # Stack along a new axis → (N, 128, 9)
    return np.stack(channels, axis=-1)


def _load_labels(split: str) -> np.ndarray:
    """Load activity labels for a split. Returns shape (N,)."""
    fpath = os.path.join(BASE_DIR, split, f"y_{split}.txt")
    return np.loadtxt(fpath, dtype=int)


def _load_subjects(split: str) -> np.ndarray:
    """Load subject IDs for a split. Returns shape (N,)."""
    fpath = os.path.join(BASE_DIR, split, f"subject_{split}.txt")
    return np.loadtxt(fpath, dtype=int)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def load_har_data() -> dict:
    """
    Load the full UCI HAR dataset from raw inertial signals.

    Returns a dictionary with keys:
        X_train, y_train, subjects_train,
        X_test,  y_test,  subjects_test,
        X, y, subjects   (combined train+test),
        channel_names, activity_labels
    """
    print("Loading UCI HAR raw inertial signals...")
    print(f"  Base directory: {BASE_DIR}\n")

    # --- Train ---
    print("  Loading TRAIN split...")
    X_train = _load_signals("train")
    y_train = _load_labels("train")
    subjects_train = _load_subjects("train")
    print(f"    X_train:        {X_train.shape}")
    print(f"    y_train:        {y_train.shape}")
    print(f"    subjects_train: {subjects_train.shape}")
    print(f"    Unique subjects (train): {np.unique(subjects_train)}")

    # --- Test ---
    print("\n  Loading TEST split...")
    X_test = _load_signals("test")
    y_test = _load_labels("test")
    subjects_test = _load_subjects("test")
    print(f"    X_test:        {X_test.shape}")
    print(f"    y_test:        {y_test.shape}")
    print(f"    subjects_test: {subjects_test.shape}")
    print(f"    Unique subjects (test):  {np.unique(subjects_test)}")

    # --- Combined ---
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    subjects = np.concatenate([subjects_train, subjects_test], axis=0)

    print("\n" + "=" * 55)
    print("  COMBINED DATASET SUMMARY")
    print("=" * 55)
    print(f"    X (signals):  {X.shape}")
    print(f"    y (labels):   {y.shape}")
    print(f"    subjects:     {subjects.shape}")
    print(f"    Unique subjects: {len(np.unique(subjects))}  →  {sorted(np.unique(subjects).tolist())}")
    print(f"    Unique activities: {sorted(np.unique(y).tolist())}")
    print("=" * 55)

    # --- Sanity checks ---
    assert X_train.shape[0] == y_train.shape[0] == subjects_train.shape[0], \
        "Train array length mismatch!"
    assert X_test.shape[0] == y_test.shape[0] == subjects_test.shape[0], \
        "Test array length mismatch!"
    assert X.shape == (10299, 128, 9), \
        f"Unexpected combined X shape: {X.shape}"
    assert len(np.unique(subjects)) == 30, \
        f"Expected 30 unique subjects, got {len(np.unique(subjects))}"
    print("\n  ✓ All sanity checks passed.\n")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "subjects_train": subjects_train,
        "X_test": X_test,
        "y_test": y_test,
        "subjects_test": subjects_test,
        "X": X,
        "y": y,
        "subjects": subjects,
        "channel_names": CHANNEL_NAMES,
        "activity_labels": ACTIVITY_LABELS,
    }


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    data = load_har_data()
