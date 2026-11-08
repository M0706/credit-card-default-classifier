"""Streamlit front-end for the credit card default classifiers.

Loads the models trained by train_models.py and scores an uploaded CSV.
No model is ever retrained on the uploaded data.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# macOS OpenBLAS raises spurious divide-by-zero warnings on matmul for denormal
# inputs; outputs are verified finite, so silence just this class of warning.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
# The scaler artefact was pickled by scikit-learn 1.6.1. Newer releases load it
# correctly and return identical predictions, but log a version warning each time.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
BUNDLED_TEST_DATA = ROOT / "test_data.csv"

META = json.loads((MODEL_DIR / "metadata.json").read_text())
FEATURES = META["features"]
TARGET = META["target"]
POSITIVE_LABEL = "Default"

MODEL_CHOICES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "mcc": "MCC",
}


@st.cache_resource
def load_model(key):
    return joblib.load(MODEL_DIR / f"{key}.joblib")


@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.joblib")


def score(model, X, y):
    proba = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X)
    )
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "auc": roc_auc_score(y, proba),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "mcc": matthews_corrcoef(y, preds),
    }, preds


def confusion_figure(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    image = ax.imshow(matrix, cmap="Blues")
    labels = ["No default", POSITIVE_LABEL]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            shade = "white" if value > matrix.max() / 2 else "#1f2933"
            ax.text(j, i, f"{value:,}", ha="center", va="center", color=shade, fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Credit Default Risk Explorer", page_icon="💳", layout="wide")

st.title("Credit Card Default Risk Explorer")
st.caption(
    "Five classifiers trained on the UCI *Default of Credit Card Clients* dataset. "
    "Upload a CSV to score it with a pre-trained model — nothing is retrained here."
)

with st.sidebar:
    st.header("Configuration")
    model_label = st.selectbox("Classifier", list(MODEL_CHOICES))
    upload = st.file_uploader("Evaluation CSV", type=["csv"])
    st.divider()
    st.caption(
        "The CSV needs the 23 input columns plus the `default` target column "
        "(1 = defaulted, 0 = did not)."
    )

if upload is not None:
    frame = pd.read_csv(upload)
    source = upload.name
else:
    frame = pd.read_csv(BUNDLED_TEST_DATA)
    source = "bundled test_data.csv"

missing = [column for column in FEATURES if column not in frame.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required column(s): {', '.join(missing[:6])}")
    st.stop()

has_target = TARGET in frame.columns
if not has_target:
    st.warning(
        f"No `{TARGET}` column found — showing predictions only, so metrics can't be computed."
    )

model = load_model(MODEL_CHOICES[model_label])
X = load_scaler().transform(frame[FEATURES])

st.subheader(f"{model_label} · scored on {source}")
st.caption(f"{len(frame):,} rows · {len(FEATURES)} features")

if has_target:
    y = frame[TARGET]
    results, preds = score(model, X, y)

    columns = st.columns(6)
    for column, (key, label) in zip(columns, METRIC_LABELS.items()):
        column.metric(label, f"{results[key]:.4f}")

    st.divider()
    matrix_column, report_column = st.columns([1, 1.15])
    with matrix_column:
        st.markdown("**Confusion matrix**")
        st.pyplot(confusion_figure(y, preds))
    with report_column:
        st.markdown("**Classification report**")
        st.dataframe(
            pd.DataFrame(
                classification_report(
                    y, preds, labels=[0, 1], target_names=["No default", POSITIVE_LABEL],
                    output_dict=True, zero_division=0,
                )
            ).transpose().round(4)
        )
        st.caption(
            "Accuracy alone is misleading: predicting 'no default' for everyone "
            f"already scores {(y == 0).mean():.4f} on this data."
        )

with st.expander("Preview uploaded data"):
    st.dataframe(frame.head(25))
    if has_target:
        st.bar_chart(frame[TARGET].value_counts().rename(index={0: "No default", 1: POSITIVE_LABEL}))
