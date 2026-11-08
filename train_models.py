"""Train the five hand-written classifiers and persist artefacts.

Every model here comes from the local `models/` package — scikit-learn is used
only for the train/test split, the scaler and the metric functions.

Run `validate_models.py` first (or after) to confirm each implementation agrees
with scikit-learn's reference version on the same data.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import (
    DecisionTreeModel,
    GaussianNBModel,
    KNNModel,
    LogisticRegressionModel,
    SVMModel,
)

DATA = Path("data/credit_default.csv")
MODEL_DIR = Path("model")
REPORT_DIR = Path("reports")
TEST_CSV = Path("test_data.csv")
TARGET = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# An n×n kernel matrix is quadratic in memory, so the SVM trains on a
# stratified 4,000-row subsample of the 23,972 training rows.
SVM_MAX_SAMPLES = 4000

warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)


def evaluate(model, X, y):
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "auc": roc_auc_score(y, scores),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "mcc": matthews_corrcoef(y, preds),
    }


def tune(name, build, grid, X_tr, y_tr, X_val, y_val):
    """Pick the configuration with the best validation F1."""
    best_params, best_score = None, float("-inf")
    for params in grid:
        model = build(**params).fit(X_tr, y_tr)
        score = f1_score(y_val, model.predict(X_val))
        print(f"  {name:14} {params}  val_f1={score:.4f}")
        if score > best_score:
            best_params, best_score = params, score
    return best_params


def build_models(tree_params, knn_params):
    return {
        "logistic_regression": LogisticRegressionModel(C=1.0),
        "decision_tree": DecisionTreeModel(**tree_params),
        "knn": KNNModel(**knn_params),
        "naive_bayes": GaussianNBModel(),
        "svm": SVMModel(C=1.0, max_samples=SVM_MAX_SAMPLES, random_state=RANDOM_STATE),
    }


def main():
    frame = pd.read_csv(DATA)
    features_frame = frame.drop(columns=[TARGET])
    feature_names = list(features_frame.columns)
    y = frame[TARGET].to_numpy(int)

    # Split row indices rather than arrays so the held-out CSV keeps its
    # original integer columns instead of being cast to float.
    train_index, test_index = train_test_split(
        np.arange(len(frame)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train = features_frame.iloc[train_index].to_numpy(float)
    X_test = features_frame.iloc[test_index].to_numpy(float)
    y_train, y_test = y[train_index], y[test_index]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    # Fit on the training features only — the test set never informs scaling.
    # Fit through a DataFrame so the scaler remembers the column names the app
    # sends at prediction time.
    scaler = StandardScaler().fit(pd.DataFrame(X_train, columns=feature_names))
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    print("Tuning the two hyperparameters that matter most:")
    tree_params = tune(
        "decision_tree",
        DecisionTreeModel,
        [{"max_depth": d, "min_samples_leaf": 50} for d in (4, 6, 8, 10, None)],
        X_tr_s, y_tr, X_val_s, y_val,
    )
    knn_params = tune(
        "knn",
        KNNModel,
        [{"n_neighbors": k, "weights": "distance"} for k in (15, 25, 45, 75)],
        X_tr_s, y_tr, X_val_s, y_val,
    )
    print(f"  chosen: decision_tree={tree_params}  knn={knn_params}\n")

    models = build_models(tree_params, knn_params)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    results, reports = {}, {}
    for name, model in models.items():
        # The tuned models above were fitted on the tuning split; every model is
        # fitted here on the full training set before being scored.
        model.fit(X_train_s, y_train)
        metrics = evaluate(model, X_test_s, y_test)
        results[name] = metrics
        preds = model.predict(X_test_s)
        reports[name] = {
            "classification_report": classification_report(y_test, preds, digits=4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        model.save(MODEL_DIR / f"{name}.joblib")
        print(f"{name:20} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps({"features": feature_names, "target": TARGET, "positive_class": 1}, indent=2)
    )

    test_out = features_frame.iloc[test_index].copy()
    test_out[TARGET] = y_test
    test_out.to_csv(TEST_CSV, index=False)

    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    (REPORT_DIR / "details.json").write_text(json.dumps(reports, indent=2))
    (REPORT_DIR / "training_config.json").write_text(
        json.dumps(
            {
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "train_rows": int(len(y_train)),
                "test_rows": int(len(y_test)),
                "decision_tree": tree_params,
                "knn": knn_params,
                "logistic_regression": {"C": 1.0},
                "naive_bayes": {"var_smoothing": 1e-9},
                "svm": {"C": 1.0, "gamma": "scale", "max_samples": SVM_MAX_SAMPLES},
            },
            indent=2,
        )
    )

    print(f"\nSaved {len(models)} models -> {MODEL_DIR}/")
    print(f"Test data ({len(test_out)} rows) -> {TEST_CSV}")


if __name__ == "__main__":
    main()
