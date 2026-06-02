# Cross-Subject Human Activity Recognition

**A comprehensive machine learning pipeline for classifying human activities from smartphone sensor data, with a focus on solving inter-subject domain shift.**

> **Live Dashboard:** [Streamlit App](https://your-app-url.streamlit.app) *(update after deployment)*

---

## Key Findings

| Model | Mean LOSO Accuracy | Std Dev | Status |
|-------|:------------------:|:-------:|--------|
| Random Forest (Baseline) | 89.88% | +/-9.11% | Baseline |
| 1D CNN (Raw Signals) | 93.30% | +/-8.63% | Best architecture |
| LSTM | 91.00% | +/-11.78% | High variance |
| 2D CNN (Spectrogram) | 62.79% | +/-10.49% | Failed |
| **1D CNN (Normalized)** | **95.76%** | **+/-5.85%** | **Winner** |

**Bottom line:** Subject-wise Z-score normalization improved the 1D CNN from 93.30% to 95.76%, while reducing variance by 32%. The worst-performing subject improved from 69.3% to 80.6%.

---

## Project Overview

This project uses the [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) containing smartphone accelerometer and gyroscope data from 30 subjects performing 6 daily activities:

- WALKING, WALKING UPSTAIRS, WALKING DOWNSTAIRS
- SITTING, STANDING, LAYING

**Evaluation protocol:** 30-fold Leave-One-Subject-Out (LOSO) cross-validation. Each fold trains on 29 subjects and tests on the 1 held-out subject. This is the gold standard for wearable sensor research because it simulates real-world deployment where the model must generalize to unseen users.

**Dataset statistics:**
- 10,299 samples (128 timesteps x 9 sensor channels)
- 30 unique subjects, 6 activity classes
- Sensor channels: total_acc (x,y,z), body_acc (x,y,z), body_gyro (x,y,z)

---

## Methodology

### Phase 1: Baseline (Random Forest)
Flattened the raw (128, 9) signals into 1,152-dimensional feature vectors and trained a RandomForestClassifier (100 trees). Achieved 89.88% mean LOSO accuracy, establishing a strong baseline.

### Phase 2A: 1D CNN
Applied a 1D Convolutional Neural Network directly on the raw (128, 9) time-series signals. Architecture: Conv1D(64) -> MaxPool -> Conv1D(128) -> MaxPool -> GlobalAvgPool -> Dense(64) -> Dense(6). Achieved 93.30%, a +3.42% improvement over RF by learning temporal patterns automatically.

### Phase 2B: LSTM
Used a single LSTM(64) layer to capture long-range temporal dependencies. Achieved 91.00% but with high variance (+/-11.78%) and required 92 minutes of training on CPU -- 9x slower than the 1D CNN for worse results.

### Phase 2C: 2D CNN on Spectrograms
Converted the first 3 channels (body_acc x,y,z) into (17, 17, 3) spectrogram images using Short-Time Fourier Transform, then applied a tiny 2D CNN. **Result: 62.79% -- a clear failure.**

**Why it failed:** The UCI HAR signal windows are only 2.56 seconds long (128 samples at 50Hz). STFTs produce meaningful spectrograms on longer recordings (10+ seconds), but on such short windows the (17, 17) images lack sufficient frequency resolution. Additionally, using only 3 of 9 channels lost 67% of the sensor information. This demonstrates that not every "modern" technique improves results -- the data characteristics must match the method's assumptions.

### Phase 3: Domain Shift Analysis and Normalization
Identified that subjects 9, 10, 14, and 16 consistently scored below 80% across all models. This inter-subject domain shift is caused by differences in walking style, body dimensions, and sensor placement.

**Fix:** Applied per-subject, per-channel Z-score normalization before training:

```
X_normalized[subject_s, :, channel_c] = (X[subject_s, :, channel_c] - mean_sc) / std_sc
```

This removes subject-specific sensor baselines, forcing the model to learn activity patterns rather than subject identity.

---

## Domain Shift Fix -- Results

The normalization fix produced the single largest accuracy improvement in the project:

| Subject | Before (Raw) | After (Normalized) | Delta | Notes |
|:-------:|:------------:|:-------------------:|:-----:|-------|
| 9 | 85.1% | 81.9% | -3.1% | Minor regression |
| 10 | 70.8% | 81.0% | +10.2% | Above 80% threshold |
| 14 | 69.3% | 93.2% | +23.8% | Largest improvement |
| 16 | 77.3% | 80.6% | +3.3% | Above 80% threshold |
| **Average** | **75.6%** | **84.2%** | **+8.5%** | |

Subject 14 had the most dramatic improvement (+23.8%), indicating its raw sensor data had a significantly different baseline distribution. After normalization, it performed on par with typical subjects.

The overall standard deviation dropped from 8.63% to 5.85% (-32%), meaning the model performs more consistently across all subjects.

---

## Error Analysis

The confusion matrix reveals one dominant error source:

```
                    Predicted
                SITTING    STANDING
True SITTING     89.6%       9.8%
True STANDING    10.1%      89.8%
```

**SITTING vs STANDING accounts for 84% of all classification errors** (366 out of 436 total errors). All other activity pairs achieve >97.8% recall.

This is a fundamental hardware limitation, not a model failure. Both activities involve the body being nearly stationary with only subtle postural differences in torso angle. Smartphone accelerometers and gyroscopes cannot reliably distinguish these postures. Solving this would require additional sensors (e.g., pressure sensors, EMG) or sensor placement on the thigh rather than the waist.

---

## Project Structure

```
UCI_HAR/
|-- app.py                          # Streamlit dashboard
|-- requirements.txt                # Python dependencies (for cloud deployment)
|-- README.md                       # This file
|
|-- data_uci_har/                   # UCI HAR Dataset
|   |-- UCI HAR Dataset/            # Raw data files
|   |-- X_spectrograms.npy          # Pre-computed spectrogram images
|
|-- src/
|   |-- data_loader.py              # Data loading and preprocessing
|   |-- loso.py                     # LOSO cross-validation generator
|   |-- models.py                   # Model architectures (1D CNN, LSTM, 2D CNN)
|   |-- normalization.py            # Subject-wise Z-score normalization
|   |-- spectrogram.py              # Signal-to-spectrogram conversion
|   |-- rf_baseline.py              # Random Forest baseline
|   |-- cnn_loso.py                 # 1D CNN LOSO evaluation
|   |-- lstm_loso.py                # LSTM LOSO evaluation
|   |-- cnn2d_loso.py               # 2D CNN LOSO evaluation
|   |-- cnn_loso_normalized.py      # 1D CNN with normalization
|   |-- error_analysis.py           # Confusion matrix prediction gathering
|   |-- plot_domain_shift.py        # Domain shift bar chart
|   |-- plot_confusion_matrix.py    # Confusion matrix visualization
|   |-- export_results.py           # CSV export for Streamlit
|
|-- results/
|   |-- per_subject_metrics.csv     # Per-subject accuracy data
|   |-- per_subject_accuracy.png    # Domain shift visualization
|   |-- confusion_matrix_normalized.png  # Confusion matrix
|   |-- rf_loso_results.txt         # RF detailed results
|   |-- cnn_loso_results.txt        # 1D CNN detailed results
|   |-- lstm_loso_results.txt       # LSTM detailed results
|   |-- cnn2d_loso_results.txt      # 2D CNN detailed results
|   |-- cnn_loso_normalized_results.txt  # Normalized CNN results
|   |-- y_true_all.npy              # Saved true labels
|   |-- y_pred_all.npy              # Saved predicted labels
```

---

## How to Run

### Prerequisites
- Python 3.10+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/UCI_HAR.git
cd UCI_HAR

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py
```

The dashboard loads pre-computed results from CSV and PNG files -- no GPU or TensorFlow required.

### Reproducing the ML Pipeline

To retrain models from scratch (requires TensorFlow):

```bash
pip install tensorflow

# Run models in order
python src/rf_baseline.py
python src/cnn_loso.py
python src/lstm_loso.py
python src/spectrogram.py          # Pre-compute spectrograms
python src/cnn2d_loso.py
python src/cnn_loso_normalized.py  # Best model
python src/error_analysis.py       # Generate predictions for confusion matrix
python src/plot_confusion_matrix.py
python src/export_results.py       # Generate CSV for dashboard
```

---

## Technologies

- **ML Framework:** TensorFlow / Keras
- **Classical ML:** scikit-learn (Random Forest)
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Evaluation:** 30-fold Leave-One-Subject-Out Cross-Validation

---

## References

- Anguita, D., et al. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN.
- UCI Machine Learning Repository: [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
