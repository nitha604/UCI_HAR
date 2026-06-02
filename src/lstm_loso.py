"""
lstm_loso.py — LSTM with Leave-One-Subject-Out Cross-Validation
=================================================================
Trains a lightweight LSTM on raw inertial signals (128, 9)
using LOSO evaluation across all 30 subjects.
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
from src.models import build_lstm_model

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──
NUM_CLASSES = 6
MAX_EPOCHS = 20
PATIENCE = 3
RF_BASELINE = 0.8988
CNN_BASELINE = 0.9330


def run_lstm_loso():
    """Run full 30-fold LOSO evaluation with LSTM."""

    # ── Load data ──
    data = load_har_data()
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    print("\n" + "=" * 65)
    print("  LSTM -- 30-FOLD LOSO EVALUATION")
    print(f"  Model: LSTM(64) -> Dropout(0.3) -> Dense(32) -> Dense(6)")
    print(f"  Params: ~21K | Epochs: {MAX_EPOCHS} max | EarlyStopping: patience={PATIENCE}")
    print("=" * 65)

    subject_accuracies = {}
    subject_epochs = {}
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    for subject_id, X_train, X_test, y_train, y_test in get_loso_splits(X, subjects, y):
        fold_start = time.time()

        # Clear session at START of each fold to prevent memory leaks
        tf.keras.backend.clear_session()

        # One-hot encode (shift 1-6 -> 0-5 first)
        y_train_oh = to_categorical(y_train - 1, num_classes=NUM_CLASSES)
        y_test_oh = to_categorical(y_test - 1, num_classes=NUM_CLASSES)

        # Build a FRESH model each fold (no weight leakage)
        model = build_lstm_model(input_shape=(128, 9), num_classes=NUM_CLASSES)
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
        y_true = y_test - 1  # shift to 0-5 for comparison

        acc = np.mean(y_pred == y_true)
        n_correct = np.sum(y_pred == y_true)
        epochs_run = len(history.history["loss"])

        subject_accuracies[subject_id] = acc
        subject_epochs[subject_id] = epochs_run
        total_correct += n_correct
        total_samples += len(y_test)

        fold_time = time.time() - fold_start
        print(f"  Fold {subject_id:2d} | Subject {subject_id:2d} | "
              f"Test: {len(y_test):4d} | "
              f"Acc: {acc:.4f} ({n_correct}/{len(y_test)}) | "
              f"Epochs: {epochs_run:2d} | "
              f"Time: {fold_time:.1f}s")

    elapsed = time.time() - start_time

    # ── Summary ──
    mean_acc = np.mean(list(subject_accuracies.values()))
    std_acc = np.std(list(subject_accuracies.values()))
    overall_acc = total_correct / total_samples
    avg_epochs = np.mean(list(subject_epochs.values()))

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY -- LSTM")
    print("=" * 65)
    print(f"  Mean subject accuracy:    {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Overall accuracy:         {overall_acc:.4f} ({total_correct}/{total_samples})")
    print(f"  Best  subject:            {max(subject_accuracies, key=subject_accuracies.get)} "
          f"({max(subject_accuracies.values()):.4f})")
    print(f"  Worst subject:            {min(subject_accuracies, key=subject_accuracies.get)} "
          f"({min(subject_accuracies.values()):.4f})")
    print(f"  Avg epochs per fold:      {avg_epochs:.1f}")
    print(f"  Total time:               {elapsed:.1f}s")
    print()
    print(f"  -- Comparison --")
    print(f"  RF  baseline:             {RF_BASELINE:.4f}")
    print(f"  CNN baseline:             {CNN_BASELINE:.4f}")
    print(f"  LSTM result:              {mean_acc:.4f}")
    delta_rf = mean_acc - RF_BASELINE
    delta_cnn = mean_acc - CNN_BASELINE
    print(f"  vs RF:                    {delta_rf:+.4f} ({'IMPROVEMENT' if delta_rf > 0 else 'REGRESSION'})")
    print(f"  vs CNN:                   {delta_cnn:+.4f} ({'IMPROVEMENT' if delta_cnn > 0 else 'REGRESSION'})")
    print("=" * 65)

    # ── Save to file ──
    save_path = os.path.join(RESULTS_DIR, "lstm_loso_results.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("LSTM -- LOSO Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Architecture: LSTM(64) -> Dropout(0.3) -> Dense(32) -> Dense(6)\n")
        f.write(f"Params: 21,222 | Max epochs: {MAX_EPOCHS} | Patience: {PATIENCE}\n\n")
        f.write(f"{'Subject':>8s}  {'Samples':>8s}  {'Accuracy':>10s}  {'Epochs':>8s}\n")
        f.write("-" * 40 + "\n")
        for sid in sorted(subject_accuracies.keys()):
            n_samples = np.sum(subjects == sid)
            f.write(f"{sid:>8d}  {n_samples:>8d}  {subject_accuracies[sid]:>10.4f}  {subject_epochs[sid]:>8d}\n")
        f.write("-" * 40 + "\n")
        f.write(f"\nMean accuracy:    {mean_acc:.4f} +/- {std_acc:.4f}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")
        f.write(f"RF baseline:      {RF_BASELINE:.4f}\n")
        f.write(f"CNN baseline:     {CNN_BASELINE:.4f}\n")
        f.write(f"vs RF:            {delta_rf:+.4f}\n")
        f.write(f"vs CNN:           {delta_cnn:+.4f}\n")

    print(f"\n  Results saved -> {save_path}\n")

    return subject_accuracies, mean_acc


if __name__ == "__main__":
    run_lstm_loso()
