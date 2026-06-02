"""
app.py -- Cross-Subject HAR Analysis Dashboard
================================================
Streamlit app visualizing the results of the UCI HAR project.
Run: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -- Page Config --
st.set_page_config(
    page_title="Cross-Subject HAR Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Paths --
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


@st.cache_data
def load_per_subject_metrics():
    """Load per-subject accuracy CSV."""
    return pd.read_csv(os.path.join(RESULTS_DIR, "per_subject_metrics.csv"))


# -- Custom CSS --
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .section-divider {
        margin: 2rem 0 1.5rem 0;
        border-top: 2px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================
#  SIDEBAR -- Subject Filter
# ==============================================================
df_all = load_per_subject_metrics()

st.sidebar.title("Filters")
st.sidebar.markdown("---")

subject_options = ["All Subjects"] + [f"Subject {s}" for s in sorted(df_all["Subject"].unique())]
selected_subject = st.sidebar.selectbox("Select Subject", subject_options)

if selected_subject == "All Subjects":
    df_filtered = df_all.copy()
    selected_id = None
else:
    selected_id = int(selected_subject.split(" ")[1])
    df_filtered = df_all[df_all["Subject"] == selected_id]

# Sidebar summary
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Summary**")
st.sidebar.markdown(f"- Subjects: {len(df_filtered)}")
st.sidebar.markdown(f"- Activities: 6")
st.sidebar.markdown(f"- Total samples: 10,299")
st.sidebar.markdown("---")
st.sidebar.markdown("**Models Evaluated**")
st.sidebar.markdown("- Random Forest")
st.sidebar.markdown("- 1D CNN (Raw)")
st.sidebar.markdown("- LSTM")
st.sidebar.markdown("- 2D CNN (Spectrogram)")
st.sidebar.markdown("- 1D CNN (Normalized)")

# ==============================================================
#  HEADER
# ==============================================================
st.markdown('<p class="main-header">Cross-Subject Human Activity Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">UCI HAR Dataset -- LOSO Evaluation Pipeline | '
            '30 Subjects - 6 Activities - 10,299 Samples</p>', unsafe_allow_html=True)

# ==============================================================
#  KPI METRICS
# ==============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Best Model",
        value="1D CNN (Norm)",
        delta="Subject-wise Z-score",
    )

with col2:
    st.metric(
        label="Mean LOSO Accuracy",
        value="95.76%",
        delta="+2.46% vs Raw CNN",
    )

with col3:
    st.metric(
        label="Std Deviation",
        value="5.85%",
        delta="-2.78% vs Raw CNN",
        delta_color="inverse",
    )

with col4:
    st.metric(
        label="Worst Subject",
        value="80.6%",
        delta="+11.3% vs Raw CNN worst",
    )

# ==============================================================
#  MODEL COMPARISON TABLE
# ==============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Model Comparison - All Architectures")

model_data = {
    "Model": [
        "Random Forest",
        "1D CNN (Raw)",
        "LSTM",
        "2D CNN (Spectrogram)",
        ">> 1D CNN (Normalized) <<",
    ],
    "Mean Accuracy": ["89.88%", "93.30%", "91.00%", "62.79%", "95.76%"],
    "Std Dev": ["+/-9.11%", "+/-8.63%", "+/-11.78%", "+/-10.49%", "+/-5.85%"],
    "Parameters": ["N/A", "36,294", "21,222", "21,670", "36,294"],
    "Training Time": ["4.4 min", "10.7 min", "92 min", "10.2 min", "7.7 min"],
    "Input": [
        "Flattened (1152,)",
        "Raw (128, 9)",
        "Raw (128, 9)",
        "Spectrogram (17,17,3)",
        "Normalized (128, 9)",
    ],
}

df_models = pd.DataFrame(model_data)

st.dataframe(
    df_models,
    width="stretch",
    hide_index=True,
    column_config={
        "Model": st.column_config.TextColumn("Model", width="large"),
        "Mean Accuracy": st.column_config.TextColumn("Mean Accuracy", width="small"),
        "Std Dev": st.column_config.TextColumn("Std Dev", width="small"),
    },
    
)

# ==============================================================
#  PROBLEM SUBJECTS TABLE
# ==============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Domain Shift Fix - Problem Subjects")

problem_data = {
    "Subject": [9, 10, 14, 16],
    "Before (Raw CNN)": ["85.1%", "70.8%", "69.3%", "77.3%"],
    "After (Norm CNN)": ["81.9%", "81.0%", "93.2%", "80.6%"],
    "Delta": ["-3.1%", "+10.2%", "+23.8%", "+3.3%"],
    "Status": ["Minor regression", "Improved", "Major fix", "Improved"],
}

df_problem = pd.DataFrame(problem_data)
st.dataframe(df_problem, width="stretch", hide_index=True)

st.info("**Key insight:** Subject-wise Z-score normalization lifted the average accuracy of problem "
        "subjects from **75.6% to 84.2%** (+8.5%), proving that inter-subject domain shift was the "
        "primary source of error.")

# ==============================================================
#  INTERACTIVE PER-SUBJECT BAR CHART (responds to sidebar filter)
# ==============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if selected_id is not None:
    st.subheader(f"Per-Subject Accuracy - Subject {selected_id}")
else:
    st.subheader("Per-Subject Accuracy Comparison")

# Melt wide -> long for Plotly grouped bars
df_melted = df_filtered.melt(
    id_vars=["Subject"],
    value_vars=["RF_Accuracy", "CNN_Raw_Accuracy", "CNN_Normalized_Accuracy"],
    var_name="Model",
    value_name="Accuracy",
)

# Clean model names for display
model_name_map = {
    "RF_Accuracy": "Random Forest",
    "CNN_Raw_Accuracy": "1D CNN (Raw)",
    "CNN_Normalized_Accuracy": "1D CNN (Normalized)",
}
df_melted["Model"] = df_melted["Model"].map(model_name_map)

# Color palette
color_map = {
    "Random Forest": "#6c757d",
    "1D CNN (Raw)": "#4a90d9",
    "1D CNN (Normalized)": "#28a745",
}

fig = px.bar(
    df_melted,
    x="Subject",
    y="Accuracy",
    color="Model",
    barmode="group",
    color_discrete_map=color_map,
    hover_data={"Accuracy": ":.1%", "Subject": True, "Model": True},
    labels={"Accuracy": "LOSO Accuracy", "Subject": "Subject ID"},
)

# 80% danger threshold line
fig.add_hline(
    y=0.80,
    line_dash="dash",
    line_color="#dc3545",
    line_width=1.5,
    annotation_text="80% threshold",
    annotation_position="top left",
    annotation_font_color="#dc3545",
)

chart_title = (f"Subject {selected_id} - LOSO Accuracy Across Models"
               if selected_id else "Per-Subject LOSO Accuracy Across Models")

fig.update_layout(
    title=dict(text=chart_title, font=dict(size=16)),
    xaxis=dict(
        title="Subject ID",
        tickmode="linear",
        dtick=1,
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        title="Accuracy",
        tickformat=".0%",
        range=[0, 1.05],
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    height=500,
    margin=dict(l=60, r=20, t=80, b=60),
    plot_bgcolor="#fafafa",
)

fig.update_traces(
    hovertemplate="<b>Subject %{x}</b><br>Accuracy: %{y:.1%}<br>Model: %{data.name}<extra></extra>"
)

st.plotly_chart(fig, width="stretch")

# Per-subject data table (also filtered)
with st.expander("View per-subject data table"):
    display_df = df_filtered.copy()
    display_df.columns = ["Subject", "RF", "1D CNN (Raw)", "1D CNN (Normalized)"]
    st.dataframe(
        display_df.style.format({
            "RF": "{:.1%}",
            "1D CNN (Raw)": "{:.1%}",
            "1D CNN (Normalized)": "{:.1%}",
        }),
        width="stretch",
        hide_index=True,
    )

# ==============================================================
#  CONFUSION MATRIX
# ==============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Global Error Analysis: SITTING vs STANDING Confusion")

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_normalized.png")
if os.path.exists(cm_path):
    col_left, col_center, col_right = st.columns([1, 3, 1])
    with col_center:
        st.image(cm_path, caption="Normalized 1D CNN - LOSO Confusion Matrix (95.8% accuracy)")

    st.markdown("""
    **Key observations from the confusion matrix:**
    - **WALKING, WALKING UPSTAIRS, WALKING DOWNSTAIRS, LAYING** are classified near-perfectly (>97.8% recall)
    - **SITTING vs STANDING** is the only significant confusion pair (~10% mutual misclassification)
    - This is a known physical limitation -- both activities produce nearly identical accelerometer signals
    - Resolving this would require gyroscope-heavy features or additional sensor modalities
    """)
else:
    st.warning("Confusion matrix image not found. Run `python src/plot_confusion_matrix.py` to generate it.")

# ==============================================================
#  DOMAIN SHIFT VISUALIZATION
# ==============================================================
ds_path = os.path.join(RESULTS_DIR, "per_subject_accuracy.png")
if os.path.exists(ds_path):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Domain Shift Visualization")

    col_left2, col_center2, col_right2 = st.columns([1, 3, 1])
    with col_center2:
        st.image(ds_path, caption="1D CNN (Raw) - Per-subject accuracy showing domain shift in problem subjects")

# ==============================================================
#  FOOTER
# ==============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("UCI HAR Dataset | 30-fold Leave-One-Subject-Out Cross-Validation | "
           "Built with TensorFlow/Keras + Streamlit")
