"""
export_results.py — Export Per-Subject Metrics to CSV for Streamlit
====================================================================
Creates a clean CSV with per-subject accuracies across RF, CNN Raw,
and CNN Normalized for use in the Streamlit dashboard.
"""

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Per-subject accuracies from LOSO runs ──

RF_ACC = {
    1: 0.9914,  2: 0.7583,  3: 0.9707,  4: 0.8265,  5: 0.8444,
    6: 0.9508,  7: 0.8636,  8: 0.9217,  9: 0.7222,  10: 0.6837,
    11: 0.9937, 12: 0.9313, 13: 0.9786, 14: 0.7152, 15: 0.9482,
    16: 0.7623, 17: 0.8424, 18: 0.9533, 19: 0.9667, 20: 0.9718,
    21: 0.8848, 22: 0.9751, 23: 0.9220, 24: 0.9764, 25: 0.8509,
    26: 0.9949, 27: 0.9920, 28: 0.9005, 29: 0.9012, 30: 0.9713,
}

CNN_RAW_ACC = {
    1:  1.0000,  2:  0.9073,  3:  1.0000,  4:  0.9558,  5:  0.8444,
    6:  0.9662,  7:  0.9578,  8:  0.9502,  9:  0.8507,  10: 0.7075,
    11: 1.0000,  12: 0.9906,  13: 0.9755,  14: 0.6935,  15: 1.0000,
    16: 0.7732,  17: 0.9538,  18: 0.9945,  19: 0.9972,  20: 1.0000,
    21: 0.9828,  22: 1.0000,  23: 0.9462,  24: 1.0000,  25: 0.8460,
    26: 0.9974,  27: 1.0000,  28: 0.8455,  29: 0.9157,  30: 0.9373,
}

CNN_NORM_ACC = {
    1:  0.9827,  2:  1.0000,  3:  0.9971,  4:  0.9401,  5:  0.8841,
    6:  0.9969,  7:  0.9805,  8:  0.9466,  9:  0.8194,  10: 0.8095,
    11: 0.9842,  12: 1.0000,  13: 1.0000,  14: 0.9319,  15: 1.0000,
    16: 0.8060,  17: 0.9647,  18: 1.0000,  19: 1.0000,  20: 0.9831,
    21: 0.9951,  22: 1.0000,  23: 0.9435,  24: 1.0000,  25: 1.0000,
    26: 0.9439,  27: 1.0000,  28: 0.8953,  29: 1.0000,  30: 0.9243,
}

# ── Build DataFrame ──
rows = []
for subject in range(1, 31):
    rows.append({
        "Subject": subject,
        "RF_Accuracy": RF_ACC[subject],
        "CNN_Raw_Accuracy": CNN_RAW_ACC[subject],
        "CNN_Normalized_Accuracy": CNN_NORM_ACC[subject],
    })

df = pd.DataFrame(rows)

# ── Save ──
csv_path = os.path.join(RESULTS_DIR, "per_subject_metrics.csv")
df.to_csv(csv_path, index=False)

print("\n" + "=" * 65)
print("  EXPORTED PER-SUBJECT METRICS")
print("=" * 65)
print(f"  Shape: {df.shape}")
print(f"  Saved: {csv_path}\n")
print(df.head(10).to_string(index=False))
print("\n  ...")
print(f"\n  Summary stats:")
print(f"    RF mean:         {df['RF_Accuracy'].mean():.4f} +/- {df['RF_Accuracy'].std():.4f}")
print(f"    CNN Raw mean:    {df['CNN_Raw_Accuracy'].mean():.4f} +/- {df['CNN_Raw_Accuracy'].std():.4f}")
print(f"    CNN Norm mean:   {df['CNN_Normalized_Accuracy'].mean():.4f} +/- {df['CNN_Normalized_Accuracy'].std():.4f}")
print("=" * 65)
