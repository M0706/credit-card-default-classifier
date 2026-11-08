"""Train five classifiers on the credit card default dataset and persist artefacts."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA = Path("data/credit_default.csv")
MODEL_DIR = Path("model")
REPORT_DIR = Path("reports")
TEST_CSV = Path("test_data.csv")
TARGET = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.2

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


def tune(name, candidates, X_tr, y_tr, X_val, y_val):
    best_score, best_model = -np.inf, None
    for params, model in candidates:
        model.fit(X_tr, y_tr)
        score = f1_score(y_val, model.predict(X_val))
        print(f"  {name:18} {params}  val_f1={score:.4f}")
        if score > best_score:
            best_score, best_model = score, model
    return best_model


def build_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    }


def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    print("Tuning the two scale-sensitive hyperparameters:")
    tree_candidates = [
        ({"max_depth": d}, DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, random_state=RANDOM_STATE))
        for d in (4, 6, 8, 10, None)
    ]
    knn_candidates = [
        ({"n_neighbors": k}, KNeighborsClassifier(n_neighbors=k, weights="distance"))
        for k in (15, 25, 45, 75)
    ]

    models = build_models()
    models["decision_tree"] = tune("decision_tree", tree_candidates, X_tr_s, y_tr, X_val_s, y_val)
    models["knn"] = tune("knn", knn_candidates, X_tr_s, y_tr, X_val_s, y_val)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    results, reports = {}, {}
    for name, model in models.items():
        # Tuned models were fitted on the tuning split; refit the chosen
        # configuration on the full training set before scoring it.
        if name in ("decision_tree", "knn"):
            model = clone(model)
        model.fit(X_train_s, y_train)
        metrics = evaluate(model, X_test_s, y_test)
        results[name] = metrics
        preds = model.predict(X_test_s)
        reports[name] = {
            "classification_report": classification_report(y_test, preds, digits=4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"{name:20} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps({"features": list(X.columns), "target": TARGET, "positive_class": 1}, indent=2)
    )

    test_out = X_test.copy()
    test_out[TARGET] = y_test.values
    test_out.to_csv(TEST_CSV, index=False)

    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    (REPORT_DIR / "details.json").write_text(json.dumps(reports, indent=2))

    print(f"\nSaved {len(models)} models -> {MODEL_DIR}/")
    print(f"Test data ({len(test_out)} rows) -> {TEST_CSV}")


if __name__ == "__main__":
    main()
