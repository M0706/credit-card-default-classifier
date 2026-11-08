"""Verify the hand-written models against scikit-learn.

Two implementations of the same algorithm should agree. This script trains both
on exactly the same split, scores both on exactly the same test set, and prints
the difference. If our maths is right the metric gaps are in the third decimal;
a real bug shows up as a large gap or a model that fails to converge.

Usage:
    python validate_models.py
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from models import (
    DecisionTreeModel,
    GaussianNBModel,
    KNNModel,
    LogisticRegressionModel,
    SVMModel,
)

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

DATA = Path("data/credit_default.csv")
REPORT = Path("reports/validation.json")
TARGET = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def scores(model, X_eval, y_true):
    """Metrics for anything that produces a probability or a decision value."""
    if hasattr(model, "predict_proba"):
        ranking = model.predict_proba(X_eval)[:, 1]
    else:
        ranking = model.decision_function(X_eval)
    return {
        "accuracy": accuracy_score(y_true, model.predict(X_eval)),
        "auc": roc_auc_score(y_true, ranking),
        "f1": f1_score(y_true, model.predict(X_eval)),
        "mcc": matthews_corrcoef(y_true, model.predict(X_eval)),
    }


def compare(label, reference, ours, X_fit, y_fit, X_eval, y_true):
    rows = []
    for name, model in (("sklearn", reference), ("from-scratch", ours)):
        started = time.perf_counter()
        model.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - started
        result = scores(model, X_eval, y_true)
        result["fit_seconds"] = elapsed
        rows.append((name, model, result))

    ref, impl = rows[0][2], rows[1][2]
    print(f"\n{label}")
    print(f"  {'metric':10} {'sklearn':>10} {'ours':>10} {'delta':>10}")
    for metric in ("accuracy", "auc", "f1", "mcc"):
        print(
            f"  {metric:10} {ref[metric]:>10.4f} {impl[metric]:>10.4f} "
            f"{impl[metric] - ref[metric]:>+10.4f}"
        )
    print(f"  fit seconds: sklearn {ref['fit_seconds']:.1f}  ours {impl['fit_seconds']:.1f}")

    entry = {"sklearn": ref, "from_scratch": impl}
    entry["max_auc_gap"] = abs(impl["auc"] - ref["auc"])
    return entry


frame = pd.read_csv(DATA)
features = [c for c in frame.columns if c != TARGET]
features_frame = frame[features]
y = frame[TARGET].to_numpy(int)

# Split row indices and fit the scaler through a DataFrame, exactly as
# train_models.py does, so the harness scores the *same* transformed matrices
# the shipped models were trained on.
train_index, test_index = train_test_split(
    np.arange(len(frame)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
X_train_frame = features_frame.iloc[train_index]
X_test_frame = features_frame.iloc[test_index]
y_train, y_test = y[train_index], y[test_index]

scaler = StandardScaler().fit(X_train_frame)
X_train_scaled = scaler.transform(X_train_frame)
X_test_scaled = scaler.transform(X_test_frame)

print(f"train {X_train_scaled.shape}  test {X_test_scaled.shape}")

results = {
    "train_rows": int(len(y_train)),
    "test_rows": int(len(y_test)),
    "models": {},
}

results["models"]["logistic_regression"] = compare(
    "Logistic Regression (Newton vs lbfgs)",
    LogisticRegression(C=1.0, max_iter=2000),
    LogisticRegressionModel(C=1.0),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)

results["models"]["decision_tree"] = compare(
    "Decision Tree (quantile thresholds vs exhaustive)",
    DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=RANDOM_STATE),
    DecisionTreeModel(max_depth=4, min_samples_leaf=50),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)

results["models"]["knn"] = compare(
    "KNN (k=15, distance weights)",
    KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
    KNNModel(n_neighbors=15, weights="distance"),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)

results["models"]["naive_bayes"] = compare(
    "Gaussian Naive Bayes",
    GaussianNB(var_smoothing=1e-9),
    GaussianNBModel(var_smoothing=1e-9),
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
)

# The SVM precomputes an n×n kernel matrix, so both implementations are fitted
# on the same stratified subsample. Same data, same comparison — just smaller.
svm_ours = SVMModel(C=1.0, max_samples=4000, random_state=RANDOM_STATE)
svm_subset, y_subset = svm_ours._subsample(X_train_scaled, y_train)
print(f"\nSVM subsample: {svm_subset.shape[0]} of {len(y_train)} training rows")

results["models"]["svm"] = compare(
    "SVM (RBF, SMO vs libsvm)",
    SVC(C=1.0, kernel="rbf", gamma="scale"),
    svm_ours,
    svm_subset,
    y_subset,
    X_test_scaled,
    y_test,
)
results["svm_subsample_rows"] = int(svm_subset.shape[0])

worst = max(entry["max_auc_gap"] for entry in results["models"].values())
results["worst_auc_gap"] = worst

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(results, indent=2))

print(f"\nlargest AUC gap vs sklearn: {worst:.4f}")
print(f"wrote {REPORT}")
