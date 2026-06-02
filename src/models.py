"""
models.py — Neural Network Architectures for UCI HAR
======================================================
Lightweight models designed for 30-fold LOSO training.
1D models: Input shape (128, 9) — 128 timesteps x 9 sensor channels.
2D models: Input shape (17, 17, 3) — spectrogram images.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pyrefly: ignore [missing-import]
import tensorflow as tf
# pyrefly: ignore [missing-import]
from tensorflow.keras import Sequential
# pyrefly: ignore [missing-import]
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, LSTM,
)


def build_cnn_model(input_shape=(128, 9), num_classes=6):
    """
    Lightweight 1D CNN for time-series classification.

    Architecture:
        Conv1D(64, k=5) → MaxPool → Conv1D(128, k=3) → MaxPool →
        GlobalAvgPool → Dropout(0.3) → Dense(64) → Dense(6, softmax)

    Estimated ~25K parameters — fast enough for 30× LOSO training.
    """
    model = Sequential([
        Conv1D(64, kernel_size=5, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def build_lstm_model(input_shape=(128, 9), num_classes=6):
    """
    Lightweight LSTM for time-series classification.

    Architecture:
        LSTM(64) -> Dropout(0.3) -> Dense(32, relu) -> Dense(6, softmax)

    Estimated ~22K parameters — fast enough for 30x LOSO training.
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    return model


def build_cnn2d_model(input_shape=(17, 17, 3), num_classes=6):
    """
    Tiny 2D CNN for spectrogram-based classification.

    Architecture:
        Conv2D(32, 3x3, same) -> MaxPool2D(2) -> Conv2D(64, 3x3, same) ->
        GlobalAvgPool2D -> Dropout(0.3) -> Dense(32) -> Dense(6, softmax)

    Only ONE pooling layer to avoid shrinking the 17x17 input to nothing.
    Estimated ~22K parameters.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    return model


# ──────────────────────────────────────────────
# Quick test -- print model summaries
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  1D CNN ARCHITECTURE")
    print("=" * 60)
    cnn = build_cnn_model()
    cnn.summary()
    print(f"\n  Total 1D CNN parameters: {cnn.count_params():,}")

    print("\n" + "=" * 60)
    print("  LSTM ARCHITECTURE")
    print("=" * 60)
    lstm = build_lstm_model()
    lstm.summary()
    print(f"\n  Total LSTM parameters: {lstm.count_params():,}")

    print("\n" + "=" * 60)
    print("  2D CNN (SPECTROGRAM) ARCHITECTURE")
    print("=" * 60)
    cnn2d = build_cnn2d_model()
    cnn2d.summary()
    print(f"\n  Total 2D CNN parameters: {cnn2d.count_params():,}")
    print("=" * 60)
