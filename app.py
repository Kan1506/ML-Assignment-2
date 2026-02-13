# app.py
# Streamlit app for ML Assignment-2 (6 models + metrics + confusion matrix)
# Dataset: Breast Cancer Wisconsin (Diagnostic) - wdbc
# Expects saved models in ./model/ and scaler in ./model/scaler.pkl

import os
import joblib
import numpy as np
import pandas as pd
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
)

import matplotlib.pyplot as plt


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="ML Assignment-2 | Classifier Demo", layout="wide")
st.title("ML Assignment-2: Classification Models Demo")
st.caption("Upload test CSV → select model → view metrics + confusion matrix (if labels provided).")


# -----------------------------
# Helpers
# -----------------------------
FEATURES_30 = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

SCALED_MODELS = {"Logistic Regression", "KNN", "Naive Bayes"}  # use scaler for these


@st.cache_resource
def load_artifacts():
    """Load scaler + all models from ./model folder."""
    base = "model"
    scaler_path = os.path.join(base, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("scaler.pkl not found inside ./model. Please save it as model/scaler.pkl")

    scaler = joblib.load(scaler_path)

    models = {}
    missing = []
    for name, fname in MODEL_FILES.items():
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            missing.append(path)
        else:
            models[name] = joblib.load(path)

    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))

    return scaler, models


def normalize_label_series(y: pd.Series) -> pd.Series:
    """
    Accepts labels like:
      - 1/0
      - M/B
      - Malignant/Benign
    Returns 1 for Malignant, 0 for Benign.
    """
    y2 = y.copy()

    if y2.dtype == object:
        y2 = y2.astype(str).str.strip().str.upper()
        y2 = y2.map({"M": 1, "B": 0, "MALIGNANT": 1, "BENIGN": 0})
    return y2.astype(int)


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df contains the 30 expected feature columns.
    - If extra columns exist (like id/diagnosis), they are ignored.
    - If any required feature is missing, raises a clear error.
    """
    missing = [c for c in FEATURES_30 if c not in df.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing required feature columns:\n"
            + ", ".join(missing)
            + "\n\nTip: Upload a CSV with the same 30 feature columns used in training."
        )
    return df[FEATURES_30].copy()


def plot_confusion(cm: np.ndarray, title: str = "Confusion Matrix"):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign (0)", "Malignant (1)"])
    ax.set_yticklabels(["Benign (0)", "Malignant (1)"])

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    st.pyplot(fig)


# -----------------------------
# Load Models
# -----------------------------
try:
    scaler, models = load_artifacts()
except Exception as e:
    st.error(str(e))
    st.stop()


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
selected_model = models[model_name]

st.sidebar.markdown("---")
st.sidebar.subheader("CSV Upload")
st.sidebar.write("Upload **test data** CSV. If it includes labels, add a column named `diagnosis`.")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

with st.sidebar.expander("Expected CSV format (recommended)"):
    st.write("Required 30 feature columns:")
    st.code(", ".join(FEATURES_30))
    st.write("Optional label column:")
    st.code("diagnosis  (values: 0/1 OR B/M)")


# -----------------------------
# Main Area
# -----------------------------
if uploaded is None:
    st.info("Upload a CSV file from the sidebar to start.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Uploaded Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# Prepare X
try:
    X = ensure_features(df)
except Exception as e:
    st.error(str(e))
    st.stop()

# Scale if needed
if model_name in SCALED_MODELS:
    X_in = scaler.transform(X)
else:
    X_in = X.values

# Predict
try:
    y_pred = selected_model.predict(X_in)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Probabilities (for AUC / confidence)
y_prob = None
if hasattr(selected_model, "predict_proba"):
    try:
        y_prob = selected_model.predict_proba(X_in)[:, 1]
    except Exception:
        y_prob = None

# Show predictions
pred_df = pd.DataFrame({"prediction": y_pred})
if y_prob is not None:
    pred_df["prob_malignant"] = y_prob

st.subheader("Predictions")
st.dataframe(pred_df.head(20), use_container_width=True)

# If labels exist, compute metrics + confusion matrix
label_col = None
for cand in ["diagnosis", "target", "label", "y"]:
    if cand in df.columns:
        label_col = cand
        break

st.markdown("---")
st.subheader("Evaluation")

if label_col is None:
    st.warning(
        "No label column found in your CSV.\n\n"
        "To compute **Accuracy/AUC/Precision/Recall/F1/MCC** and show a **Confusion Matrix**, "
        "include a label column named `diagnosis` (values: 0/1 or B/M) in the uploaded CSV."
    )
    st.write("You can still use the app to generate predictions using the selected model.")
else:
    try:
        y_true = normalize_label_series(df[label_col])
    except Exception as e:
        st.error(f"Could not parse label column `{label_col}`: {e}")
        st.stop()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1 Score", f"{f1:.4f}")
    c5.metric("MCC", f"{mcc:.4f}")
    c6.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")

    # Confusion Matrix + Classification Report
    cm = confusion_matrix(y_true, y_pred)
    st.subheader("Confusion Matrix")
    plot_confusion(cm)

    with st.expander("Classification Report"):
        st.text(classification_report(y_true, y_pred, target_names=["Benign (0)", "Malignant (1)"]))

st.markdown("---")
st.caption("Tip: For full assignment marks, deploy this app on Streamlit Cloud and include the live link in your submission.")
