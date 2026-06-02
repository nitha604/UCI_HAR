"""
cnn_loso_normalized.py — 1D CNN with Subject-Wise Normalization + LOSO
========================================================================
Re-runs the 1D CNN LOSO evaluation on normalized data (X_norm) to test
whether subject-wise z-score normalization reduces domain shift and
improves accuracy on problem subjects (9, 10, 14, 16).
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
from src.data_loader import load_har_data
from src.loso import get_loso_splits
from src.models import build_cnn_model
from src.normalization import normalize_subject_wise

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──
NUM_CLASSES = 6
MAX_EPOCHS = 20
PATIENCE = 3

# Baseline accuracies from raw-data CNN run (for comparison)
RAW_CNN_MEAN = 0.9330
RAW_CNN_STD = 0.0863
RAW_CNN_PER_SUBJECT = {
    1:  1.0000,  2:  0.9073,  3:  1.0000,  4:  0.9558,  5:  0.8444,
    6:  0.9662,  7:  0.9578,  8:  0.9502,  9:  0.8507,  10: 0.7075,
    11: 1.0000,  12: 0.9906,  13: 0.9755,  14: 0.6935,  15: 1.0000,
    16: 0.7732,  17: 0.9538,  18: 0.9945,  19: 0.9972,  20: 1.0000,
    21: 0.9828,  22: 1.0000,  23: 0.9462,  24: 1.0000,  25: 0.8460,
    26: 0.9974,  27: 1.0000,  28: 0.8455,  29: 0.9157,  30: 0.9373,
}
PROBLEM_SUBJECTS = [9, 10, 14, 16]


def run_cnn_loso_normalized():
    """Run full 30-fold LOSO evaluation with 1D CNN on normalized data."""

    # ── Load data ──
    data = load_har_data()
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    # ── Apply subject-wise normalization ──
    print("\n  Applying subject-wise Z-score normalization...")
    X_norm = normalize_subject_wise(X, subjects)
    print(f"  Done! X_norm shape: {X_norm.shape}")

    print("\n" + "=" * 70)
    print("  1D CNN (NORMALIZED) -- 30-FOLD LOSO EVALUATION")
    print(f"  Model: Conv1D(64)->MaxPool->Conv1D(128)->MaxPool->GAP->Dense(64)->Dense(6)")
    print(f"  Preprocessing: Subject-wise Z-score normalization")
    print(f"  Epochs: {MAX_EPOCHS} max | EarlyStopping: patience={PATIENCE}")
    print("=" * 70)

    subject_accuracies = {}
    subject_epochs = {}
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    for subject_id, X_train, X_test, y_train, y_test in get_loso_splits(X_norm, subjects, y):
        fold_start = time.time()

        # Clear session at START of each fold
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

        # Train with early stopping
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )

        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_test, y_test_oh),
            epochs=MAX_EPOCHS,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0,
        )

        # Predict
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = y_test - 1

        acc = np.mean(y_pred == y_true)
        n_correct = np.sum(y_pred == y_true)
        epochs_run = len(history.history["loss"])

        subject_accuracies[subject_id] = acc
        subject_epochs[subject_id] = epochs_run
        total_correct += n_correct
        total_samples += len(y_test)

        fold_time = time.time() - fold_start

        # Mark problem subjects
        tag = " *** PROBLEM" if subject_id in PROBLEM_SUBJECTS else ""
        print(f"  Fold {subject_id:2d} | Subject {subject_id:2d} | "
              f"Test: {len(y_test):4d} | "
              f"Acc: {acc:.4f} ({n_correct}/{len(y_test)}) | "
              f"Epochs: {epochs_run:2d} | "
              f"Time: {fold_time:.1f}s{tag}")

    elapsed = time.time() - start_time

    # ── Overall Summary ──
    mean_acc = np.mean(list(subject_accuracies.values()))
    std_acc = np.std(list(subject_accuracies.values()))
    overall_acc = total_correct / total_samples
    avg_epochs = np.mean(list(subject_epochs.values()))

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY -- 1D CNN (NORMALIZED)")
    print("=" * 70)
    print(f"  Mean subject accuracy:    {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Overall accuracy:         {overall_acc:.4f} ({total_correct}/{total_samples})")
    print(f"  Best  subject:            {max(subject_accuracies, key=subject_accuracies.get)} "
          f"({max(subject_accuracies.values()):.4f})")
    print(f"  Worst subject:            {min(subject_accuracies, key=subject_accuracies.get)} "
          f"({min(subject_accuracies.values()):.4f})")
    print(f"  Avg epochs per fold:      {avg_epochs:.1f}")
    print(f"  Total time:               {elapsed:.1f}s")

    # ── Overall Before vs After ──
    print("\n" + "=" * 70)
    print("  OVERALL COMPARISON -- RAW vs NORMALIZED")
    print("=" * 70)
    print(f"  {'Metric':<25s}  {'Raw CNN':>12s}  {'Norm CNN':>12s}  {'Delta':>10s}")
    print("  " + "-" * 62)
    delta_mean = mean_acc - RAW_CNN_MEAN
    delta_std = std_acc - RAW_CNN_STD
    print(f"  {'Mean accuracy':<25s}  {RAW_CNN_MEAN:>12.4f}  {mean_acc:>12.4f}  {delta_mean:>+10.4f}")
    print(f"  {'Std deviation':<25s}  {RAW_CNN_STD:>12.4f}  {std_acc:>12.4f}  {delta_std:>+10.4f}")

    # ══════════════════════════════════════════════════════════
    # THE KEY TABLE: Problem Subjects Before vs After
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PROBLEM SUBJECTS -- BEFORE vs AFTER NORMALIZATION")
    print("  (This is the key domain shift result)")
    print("=" * 70)
    print(f"  {'Subject':>8s}  {'Before (raw)':>14s}  {'After (norm)':>14s}  {'Delta':>10s}  {'Status':>10s}")
    print("  " + "-" * 60)
    for sid in PROBLEM_SUBJECTS:
        before = RAW_CNN_PER_SUBJECT[sid]
        after = subject_accuracies[sid]
        delta = after - before
        status = "IMPROVED" if delta > 0 else "WORSE"
        print(f"  {sid:>8d}  {before*100:>13.1f}%  {after*100:>13.1f}%  {delta*100:>+9.1f}%  {status:>10s}")

    # Average improvement on problem subjects
    avg_before = np.mean([RAW_CNN_PER_SUBJECT[s] for s in PROBLEM_SUBJECTS])
    avg_after = np.mean([subject_accuracies[s] for s in PROBLEM_SUBJECTS])
    print("  " + "-" * 60)
    print(f"  {'AVG':>8s}  {avg_before*100:>13.1f}%  {avg_after*100:>13.1f}%  {(avg_after-avg_before)*100:>+9.1f}%")
    print("=" * 70)

    # ── Save to file ──
    save_path = os.path.join(RESULTS_DIR, "cnn_loso_normalized_results.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("1D CNN (Normalized) -- LOSO Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Preprocessing: Subject-wise Z-score normalization\n")
        f.write(f"Architecture: Conv1D(64)->MaxPool->Conv1D(128)->MaxPool->GAP->Dense(64)->Dense(6)\n")
        f.write(f"Params: 36,294 | Max epochs: {MAX_EPOCHS} | Patience: {PATIENCE}\n\n")
        f.write(f"{'Subject':>8s}  {'Samples':>8s}  {'Raw Acc':>10s}  {'Norm Acc':>10s}  {'Delta':>10s}\n")
        f.write("-" * 50 + "\n")
        for sid in sorted(subject_accuracies.keys()):
            n_samples = np.sum(subjects == sid)
            raw = RAW_CNN_PER_SUBJECT[sid]
            norm = subject_accuracies[sid]
            f.write(f"{sid:>8d}  {n_samples:>8d}  {raw:>10.4f}  {norm:>10.4f}  {norm-raw:>+10.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"\nRaw CNN mean:        {RAW_CNN_MEAN:.4f} +/- {RAW_CNN_STD:.4f}\n")
        f.write(f"Normalized CNN mean: {mean_acc:.4f} +/- {std_acc:.4f}\n")
        f.write(f"Delta:               {delta_mean:+.4f}\n")

    print(f"\n  Results saved -> {save_path}\n")

    return subject_accuracies, mean_acc


if __name__ == "__main__":
    run_cnn_loso_normalized()
