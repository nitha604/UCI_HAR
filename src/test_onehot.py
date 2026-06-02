"""
test_onehot.py — Quick test: one-hot encoding for Keras
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_har_data

# Suppress TF info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical

data = load_har_data()
y = data["y"]

print("\n" + "=" * 55)
print("  ONE-HOT ENCODING TEST")
print("=" * 55)
print(f"  Original y shape: {y.shape}")
print(f"  Original y range: {y.min()} – {y.max()}")

# Shift 1-6 → 0-5, then one-hot
y_zero = y - 1
y_onehot = to_categorical(y_zero, num_classes=6)

print(f"\n  y_zero range:     {y_zero.min()} – {y_zero.max()}")
print(f"  y_onehot shape:   {y_onehot.shape}")
print(f"  y_onehot dtype:   {y_onehot.dtype}")

# Show a few examples
print("\n  Sample verification:")
for i in [0, 100, 5000]:
    print(f"    y[{i}] = {y[i]}  →  y_zero = {y_zero[i]}  →  one-hot = {y_onehot[i].astype(int).tolist()}")

print("=" * 55)
