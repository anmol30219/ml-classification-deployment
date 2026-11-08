"""
=============================================================================
Course: BITS Pilani M.Tech (AIML) - Machine Learning
Assignment: Assignment 2 (Streamlit Interactive Cloud Application)
Author: Anmol Ahlawat
Module: app.py
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Page Configuration & Modern Clean Layout
st.set_page_config(
    page_title="Breast Cancer Diagnostics ML Suite | BITS Pilani",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions ----------------- #
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

@st.cache_resource
def load_all_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"Model artifact not found: {path}. Please run train.py first.")
    return models

def calculate_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

# ----------------- Application Header ----------------- #
st.markdown('<div class="main-title">🔬 Clinical Diagnostic ML Decision Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>Author:</b> Anmol Ahlawat | <b>Course:</b> BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2</div>', unsafe_allow_html=True)
st.divider()

# ----------------- Sidebar Controls ----------------- #
st.sidebar.header("🕹️ Control Panel")
uploaded_file = st.sidebar.file_uploader(
    "1. Upload Test CSV (with target column)",
    type=["csv"],
    help="Upload the test dataset. If left blank, the pre-packaged 'test_data.csv' is loaded automatically."
)

selected_model_option = st.sidebar.selectbox(
    "2. Select Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine", "⚡ Compare All 5 Models"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset Specs:**
- **Repository:** UCI Machine Learning
- **Task:** Wisconsin Breast Cancer Diagnosis
- **Classes:** 0 (Benign), 1 (Malignant)
- **Features:** 30 Continuous Nuclear Attributes
""")

# ----------------- Data Loading Logic ----------------- #
test_df = None
if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded: {test_df.shape[0]} rows, {test_df.shape[1]} cols")
    except Exception as e:
        st.sidebar.error(f"Error parsing uploaded CSV: {e}")
elif os.path.exists("test_data.csv"):
    test_df = pd.read_csv("test_data.csv")
    st.sidebar.warning("Using pre-packaged `test_data.csv` (Holdout Set: 114 instances).")
else:
    st.error("No test data found. Please upload a CSV file containing input features and a 'target' column.")
    st.stop()

# Ensure Target Exists
target_col_candidates = [col for col in ['target', 'diagnosis', 'label', 'Class'] if col in test_df.columns]
if not target_col_candidates:
    st.error("Input CSV must contain a valid target column named 'target', 'diagnosis', or 'label'.")
    st.stop()
target_col = target_col_candidates[0]

X_eval = test_df.drop(columns=[target_col])
y_eval = test_df[target_col].astype(int)

# Load Trained Model Pipelines
models = load_all_models()
if len(models) < 5:
    st.error("One or more trained model artifacts are missing from './model/'. Please verify repository contents.")
    st.stop()

# ----------------- Execution Mode: Single Model ----------------- #
if selected_model_option != "⚡ Compare All 5 Models":
    clf_pipeline = models[selected_model_option]
    
    # Inference without re-training
    y_pred = clf_pipeline.predict(X_eval)
    y_proba = clf_pipeline.predict_proba(X_eval)[:, 1] if hasattr(clf_pipeline, "predict_proba") else None
    
    metrics = calculate_metrics(y_eval, y_pred, y_proba)
    
    st.subheader(f"📊 Performance Metrics: **{selected_model_option}**")
    
    # Metric KPI Display Cards
    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    cols[1].metric("AUC Score", f"{metrics['AUC']:.4f}")
    cols[2].metric("Precision", f"{metrics['Precision']:.4f}")
    cols[3].metric("Recall", f"{metrics['Recall']:.4f}")
    cols[4].metric("F1 Score", f"{metrics['F1']:.4f}")
    cols[5].metric("MCC Score", f"{metrics['MCC']:.4f}")
    
    st.write("")
    
    # Visual Analytics: Confusion Matrix & ROC Curve
    vis_col1, vis_col2 = st.columns([1, 1])
    
    with vis_col1:
        st.markdown("#### **Confusion Matrix**")
        cm = confusion_matrix(y_eval, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Benign (0)', 'Malignant (1)'],
                    yticklabels=['Benign (0)', 'Malignant (1)'],
                    ax=ax_cm, annot_kws={"size": 12, "weight": "bold"})
        ax_cm.set_xlabel("Predicted Diagnosis", fontsize=10, fontweight="bold")
        ax_cm.set_ylabel("True Ground Truth", fontsize=10, fontweight="bold")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with vis_col2:
        st.markdown("#### **ROC Curve**")
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_eval, y_proba)
            fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.8))
            ax_roc.plot(fpr, tpr, color='#1E3A8A', lw=2.5, label=f"ROC (AUC = {metrics['AUC']:.4f})")
            ax_roc.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.5, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
            ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
            ax_roc.grid(True, linestyle=':', alpha=0.6)
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            plt.close(fig_roc)
        else:
            st.info("Probability estimation is not supported for this model.")

    # Detailed Classification Report
    st.markdown("#### **Detailed Classification Report**")
    cr_dict = classification_report(y_eval, y_pred, target_names=['Benign (0)', 'Malignant (1)'], output_dict=True)
    cr_df = pd.DataFrame(cr_dict).transpose()
    st.dataframe(cr_df.style.format(precision=4).background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]))

# ----------------- Execution Mode: Compare All 5 Models ----------------- #
else:
    st.subheader("⚡ Comprehensive Multi-Model Benchmarking")
    comparison_records = []
    roc_dict = {}

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_eval)
        y_proba = pipeline.predict_proba(X_eval)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        m = calculate_metrics(y_eval, y_pred, y_proba)
        comparison_records.append({
            "ML Model Name": name,
            "Accuracy": f"{m['Accuracy']:.4f}",
            "AUC": f"{m['AUC']:.4f}",
            "Precision": f"{m['Precision']:.4f}",
            "Recall": f"{m['Recall']:.4f}",
            "F1": f"{m['F1']:.4f}",
            "MCC": f"{m['MCC']:.4f}"
        })
        
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_eval, y_proba)
            roc_dict[name] = (fpr, tpr, m['AUC'])

    comp_df = pd.DataFrame(comparison_records)
    st.markdown("##### **Model Comparison Table**")
    st.dataframe(comp_df, use_container_width=True)

    # Multi-ROC Overlay Plot
    st.markdown("##### **Superimposed ROC Curves**")
    fig_multi, ax_multi = plt.subplots(figsize=(7, 4.2))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (name, (fpr, tpr, auc_val)) in enumerate(roc_dict.items()):
        ax_multi.plot(fpr, tpr, color=colors[idx], lw=2.0, label=f"{name} (AUC = {auc_val:.4f})")
    ax_multi.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.2)
    ax_multi.set_xlabel("False Positive Rate", fontsize=10)
    ax_multi.set_ylabel("True Positive Rate", fontsize=10)
    ax_multi.legend(loc="lower right", fontsize=9)
    ax_multi.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig_multi)
    plt.close(fig_multi)

# ----------------- Data Sample Preview ----------------- #
with st.expander("🔍 Inspect Uploaded Test Data Sample"):
    st.dataframe(test_df.head(10))