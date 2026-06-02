"""
spectrogram.py — Signal-to-Image Pipeline for UCI HAR
=======================================================
Converts raw 1D accelerometer signals into 2D spectrograms
using Short-Time Fourier Transform (STFT).

Input:  (128,) single-channel signal @ 50 Hz
Output: (17, 13) magnitude spectrogram (freq_bins x time_segments)

Stacking 3 channels (body_acc_x/y/z) gives a (17, 13, 3) "image".
"""

import sys
import os
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_har_data, CHANNEL_NAMES, ACTIVITY_LABELS

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── STFT Parameters ──
FS = 50            # Sampling rate: 50 Hz
NPERSEG = 32       # Window length: 32 samples (0.64s)
NOVERLAP = 24      # 75% overlap
# Body accelerometer channels (indices 3, 4, 5 in our channel order)
BODY_ACC_CHANNELS = [3, 4, 5]
BODY_ACC_NAMES = ["body_acc_x", "body_acc_y", "body_acc_z"]


def signal_to_spectrogram(signal_1d, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP):
    """
    Convert a single 1D signal (128 timesteps) into a magnitude spectrogram.

    Args:
        signal_1d: np.ndarray of shape (128,)
        fs: sampling frequency (Hz)
        nperseg: STFT window length
        noverlap: STFT overlap

    Returns:
        np.ndarray of shape (freq_bins, time_segments) = (17, 13)
        Magnitude spectrogram (|STFT|).
    """
    freqs, times, Zxx = stft(signal_1d, fs=fs, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    return magnitude, freqs, times


def sample_to_3ch_spectrogram(sample, channels=BODY_ACC_CHANNELS):
    """
    Convert a single (128, 9) sample into a 3-channel spectrogram image.

    Args:
        sample: np.ndarray of shape (128, 9)
        channels: list of 3 channel indices to use

    Returns:
        np.ndarray of shape (freq_bins, time_segments, 3) = (17, 17, 3)
    """
    specs = []
    for ch_idx in channels:
        mag, _, _ = signal_to_spectrogram(sample[:, ch_idx])
        specs.append(mag)
    return np.stack(specs, axis=-1)


def precompute_spectrograms(X, channels=BODY_ACC_CHANNELS, save_path=None):
    """
    Pre-compute 3-channel spectrograms for ALL samples.

    Args:
        X: np.ndarray of shape (N, 128, 9)
        channels: list of 3 channel indices
        save_path: if provided, save the result as .npy

    Returns:
        np.ndarray of shape (N, freq_bins, time_segments, 3)
    """
    import time as _time

    n_samples = X.shape[0]
    print(f"\n  Pre-computing spectrograms for {n_samples} samples...")
    print(f"  Channels: {[CHANNEL_NAMES[c] for c in channels]}")

    start = _time.time()

    # Get output shape from first sample
    test_spec = sample_to_3ch_spectrogram(X[0], channels)
    freq_bins, time_bins, n_channels = test_spec.shape

    # Pre-allocate array
    X_specs = np.empty((n_samples, freq_bins, time_bins, n_channels), dtype=np.float32)
    X_specs[0] = test_spec

    for i in range(1, n_samples):
        X_specs[i] = sample_to_3ch_spectrogram(X[i], channels)
        if (i + 1) % 2000 == 0:
            print(f"    Processed {i + 1}/{n_samples}...")

    elapsed = _time.time() - start
    print(f"    Done! {elapsed:.1f}s")
    print(f"    Shape: {X_specs.shape}")
    print(f"    Dtype: {X_specs.dtype}")
    print(f"    Size:  {X_specs.nbytes / 1e6:.1f} MB")

    if save_path:
        np.save(save_path, X_specs)
        print(f"    Saved -> {save_path}")

    return X_specs


# ──────────────────────────────────────────────
# Main: test + precompute
# ──────────────────────────────────────────────
if __name__ == "__main__":
    data = load_har_data()
    X = data["X"]
    y = data["y"]
    subjects = data["subjects"]

    # Pick sample 0
    sample = X[0]  # shape: (128, 9)
    activity = ACTIVITY_LABELS[y[0]]
    subject = subjects[0]

    print("\n" + "=" * 55)
    print("  SPECTROGRAM TEST -- SINGLE SAMPLE")
    print("=" * 55)
    print(f"  Sample index: 0")
    print(f"  Subject: {subject} | Activity: {activity}")
    print(f"  Raw sample shape: {sample.shape}")

    # Single channel test
    mag, freqs, times = signal_to_spectrogram(sample[:, 3])  # body_acc_x
    print(f"\n  Single-channel spectrogram:")
    print(f"    Frequency bins: {len(freqs)}  (0 to {freqs[-1]:.1f} Hz)")
    print(f"    Time segments:  {len(times)}  (0 to {times[-1]:.3f} s)")
    print(f"    Magnitude shape: {mag.shape}")

    # 3-channel test
    spec_3ch = sample_to_3ch_spectrogram(sample)
    print(f"\n  3-channel spectrogram image:")
    print(f"    Shape: {spec_3ch.shape}  (freq x time x channels)")
    print("=" * 55)

    # ── Plot the 3 individual spectrograms ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = ["Reds", "Blues", "Greens"]

    for i, (ch_idx, ch_name, cmap) in enumerate(zip(BODY_ACC_CHANNELS, BODY_ACC_NAMES, colors)):
        mag_i, freqs_i, times_i = signal_to_spectrogram(sample[:, ch_idx])
        im = axes[i].pcolormesh(times_i, freqs_i, mag_i, shading="gouraud", cmap=cmap)
        axes[i].set_title(f"{ch_name}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Time (s)")
        if i == 0:
            axes[i].set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=axes[i], label="|STFT|")

    fig.suptitle(
        f"STFT Spectrograms -- Subject {subject}, {activity} (sample 0)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "spectrogram_sample0.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved -> {save_path}")
    plt.close()

    # ── Pre-compute ALL spectrograms ──
    print("\n" + "=" * 55)
    print("  PRE-COMPUTING FULL SPECTROGRAM DATASET")
    print("=" * 55)

    npy_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_uci_har", "X_spectrograms.npy",
    )
    X_specs = precompute_spectrograms(X, save_path=npy_path)

    print("=" * 55)

