"""
cnn2d_loso.py — 2D CNN on Spectrograms with LOSO Cross-Validation
===================================================================
Trains a tiny 2D CNN on pre-computed spectrogram images (17, 17, 3)
using Leave-One-Subject-Out evaluation across all 30 subjects.
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
from src.models import build_cnn2d_model

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SPEC_PATH = os.path.join(PROJECT_ROOT, "data_uci_har", "X_spectrograms.npy")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ──
NUM_CLASSES = 6
MAX_EPOCHS = 20
PATIENCE = 3
RF_BASELINE = 0.8988
CNN1D_BASELINE = 0.9330
LSTM_BASELINE = 0.9100


def run_cnn2d_loso():
    """Run full 30-fold LOSO evaluation with 2D CNN on spectrograms."""

    # ── Load pre-computed spectrograms ──
    print("\n  Loading pre-computed spectrograms...")
    X_specs = np.load(SPEC_PATH)
    print(f"    X_spectrograms shape: {X_specs.shape}")
    print(f"    Dtype: {X_specs.dtype} | Size: {X_specs.nbytes / 1e6:.1f} MB")

    # ── Load labels and subjects ──
    data = load_har_data()
    y = data["y"]
    subjects = data["subjects"]

    # Verify alignment
    assert X_specs.shape[0] == len(y) == len(subjects), "Array length mismatch!"
    print(f"    Alignment check: {X_specs.shape[0]} samples OK")

    print("\n" + "=" * 65)
    print("  2D CNN (SPECTROGRAM) -- 30-FOLD LOSO EVALUATION")
    print(f"  Model: Conv2D(32)->MaxPool->Conv2D(64)->GAP->Dense(32)->Dense(6)")
    print(f"  Input: (17, 17, 3) spectrogram | Params: ~21.7K")
    print(f"  Epochs: {MAX_EPOCHS} max | EarlyStopping: patience={PATIENCE}")
    print("=" * 65)

    subject_accuracies = {}
    subject_epochs = {}
    total_correct = 0
    total_samples = 0
    unique_subjects = sorted(np.unique(subjects))
    start_time = time.time()

    for subject_id in unique_subjects:
        fold_start = time.time()

        # Clear session at START of each fold
        tf.keras.backend.clear_session()

        # Split by subject
        test_mask = (subjects == subject_id)
        train_mask = ~test_mask

        X_train = X_specs[train_mask]
        X_test = X_specs[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        # One-hot encode (shift 1-6 -> 0-5)
        y_train_oh = to_categorical(y_train - 1, num_classes=NUM_CLASSES)
        y_test_oh = to_categorical(y_test - 1, num_classes=NUM_CLASSES)

        # Build fresh model
        model = build_cnn2d_model(input_shape=(17, 17, 3), num_classes=NUM_CLASSES)
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
    print("  RESULTS SUMMARY -- 2D CNN (SPECTROGRAM)")
    print("=" * 65)
    print(f"  Mean subject accuracy:    {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Overall accuracy:         {overall_acc:.4f} ({total_correct}/{total_samples})")
    print(f"  Best  subject:            {max(subject_accuracies, key=subject_accuracies.get)} "
          f"({max(subject_accuracies.values()):.4f})")
    print(f"  Worst subject:            {min(subject_accuracies, key=subject_accuracies.get)} "
          f"({min(subject_accuracies.values()):.4f})")
    print(f"  Avg epochs per fold:      {avg_epochs:.1f}")
    print(f"  Total time:               {elapsed:.1f}s")

    # ── Master comparison ──
    print("\n" + "=" * 65)
    print("  MASTER COMPARISON -- ALL MODELS")
    print("=" * 65)
    print(f"  {'Model':<25s}  {'Mean Acc':>10s}  {'Std':>8s}  {'Status':>12s}")
    print("  " + "-" * 58)
    models = [
        ("Random Forest", RF_BASELINE, 0.0911),
        ("1D CNN", CNN1D_BASELINE, 0.0863),
        ("LSTM", LSTM_BASELINE, 0.1178),
        ("2D CNN (Spectrogram)", mean_acc, std_acc),
    ]
    best_acc = max(m[1] for m in models)
    for name, acc, std in models:
        tag = "<-- BEST" if acc == best_acc else ""
        print(f"  {name:<25s}  {acc:>10.4f}  {std:>8.4f}  {tag:>12s}")
    print("=" * 65)

    # ── Save to file ──
    save_path = os.path.join(RESULTS_DIR, "cnn2d_loso_results.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("2D CNN (Spectrogram) -- LOSO Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Architecture: Conv2D(32)->MaxPool->Conv2D(64)->GAP->Dense(32)->Dense(6)\n")
        f.write(f"Input: (17, 17, 3) spectrogram images\n")
        f.write(f"Params: 21,670 | Max epochs: {MAX_EPOCHS} | Patience: {PATIENCE}\n\n")
        f.write(f"{'Subject':>8s}  {'Samples':>8s}  {'Accuracy':>10s}  {'Epochs':>8s}\n")
        f.write("-" * 40 + "\n")
        for sid in sorted(subject_accuracies.keys()):
            n_samples = np.sum(subjects == sid)
            f.write(f"{sid:>8d}  {n_samples:>8d}  {subject_accuracies[sid]:>10.4f}  {subject_epochs[sid]:>8d}\n")
        f.write("-" * 40 + "\n")
        f.write(f"\nMean accuracy:    {mean_acc:.4f} +/- {std_acc:.4f}\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n\n")
        f.write(f"--- Baselines ---\n")
        f.write(f"RF:       {RF_BASELINE:.4f}\n")
        f.write(f"1D CNN:   {CNN1D_BASELINE:.4f}\n")
        f.write(f"LSTM:     {LSTM_BASELINE:.4f}\n")
        f.write(f"2D CNN:   {mean_acc:.4f}\n")

    print(f"\n  Results saved -> {save_path}\n")

    return subject_accuracies, mean_acc


if __name__ == "__main__":
    run_cnn2d_loso()
