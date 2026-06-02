"""
error_analysis.py — Confusion Matrix & Error Analysis (Normalized 1D CNN)
==========================================================================
Re-runs the 30-fold LOSO loop on normalized data to collect ALL predictions,
then generates a confusion matrix to analyze inter-activity errors.
"""

import sys
import os
import time
import numpy as np

# Suppress TF verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pyrefly: ignore [missing-import]
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical
# pyrefly: ignore [missing-import]
from tensorflow.keras.callbacks import EarlyStopping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_har_data, ACTIVITY_LABELS
from src.loso import get_loso_splits
from src.models import build_cnn_model
from src.normalization import normalize_subject_wise

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──
NUM_CLASSES = 6
MAX_EPOCHS = 20
PATIENCE = 3


def gather_predictions():
    """
    Run 30-fold LOSO on normalized data and collect all true/predicted labels.

    Returns:
        y_true_all: np.ndarray of shape (10299,) — true labels (1-6)
        y_pred_all: np.ndarray of shape (10299,) — predicted labels (1-6)
    """

    # ── Load and normalize ──
    data = load_har_data()
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    print("\n  Applying subject-wise Z-score normalization...")
    X_norm = normalize_subject_wise(X, subjects)

    print("\n" + "=" * 65)
    print("  GATHERING PREDICTIONS -- 30-FOLD LOSO (NORMALIZED 1D CNN)")
    print("=" * 65)

    y_true_all = []
    y_pred_all = []
    start_time = time.time()

    for subject_id, X_train, X_test, y_train, y_test in get_loso_splits(X_norm, subjects, y):
        fold_start = time.time()

        tf.keras.backend.clear_session()

        # One-hot encode (shift 1-6 -> 0-5)
        y_train_oh = to_categorical(y_train - 1, num_classes=NUM_CLASSES)
        y_test_oh = to_categorical(y_test - 1, num_classes=NUM_CLASSES)

        # Build fresh model
        model = build_cnn_model(input_shape=(128, 9), num_classes=NUM_CLASSES)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )

        model.fit(
            X_train, y_train_oh,
            validation_data=(X_test, y_test_oh),
            epochs=MAX_EPOCHS,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0,
        )

        # Predict
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1) + 1  # shift back to 1-6

        # Append to master lists
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        fold_time = time.time() - fold_start
        acc = np.mean(y_pred == y_test)
        print(f"  Fold {subject_id:2d} | Acc: {acc:.4f} | Time: {fold_time:.1f}s")

    elapsed = time.time() - start_time

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # ── Verify ──
    overall_acc = np.mean(y_true_all == y_pred_all)
    print("\n" + "=" * 65)
    print("  PREDICTION GATHERING COMPLETE")
    print("=" * 65)
    print(f"  y_true_all length: {len(y_true_all)}")
    print(f"  y_pred_all length: {len(y_pred_all)}")
    print(f"  Overall accuracy:  {overall_acc:.4f} ({np.sum(y_true_all == y_pred_all)}/{len(y_true_all)})")
    print(f"  Total time:        {elapsed:.1f}s")
    print(f"  Unique true:       {sorted(np.unique(y_true_all))}")
    print(f"  Unique pred:       {sorted(np.unique(y_pred_all))}")
    print("=" * 65)

    # ── Save predictions for future analysis ──
    np.save(os.path.join(RESULTS_DIR, "y_true_all.npy"), y_true_all)
    np.save(os.path.join(RESULTS_DIR, "y_pred_all.npy"), y_pred_all)
    print(f"\n  Saved y_true_all.npy and y_pred_all.npy to results/")

    return y_true_all, y_pred_all


if __name__ == "__main__":
    y_true, y_pred = gather_predictions()
