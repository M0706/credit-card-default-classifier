"""Supporting analysis behind the numbers quoted in the README.

1. Bootstrap confidence intervals — are the differences between models real, or
   just sampling noise on a 5,993-row test set?
2. Threshold sweep — the default 0.5 cut-off is arbitrary; where does F1 peak?
3. Decision-tree feature importances — which columns actually drive the split?

Writes reports/analysis.json and prints a summary.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path("model")
REPORT_DIR = Path("reports")
TEST_CSV = Path("test_data.csv")
TARGET = "default"
N_BOOTSTRAP = 300
SEED = 42

MODEL_KEYS = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "svm",
]

# macOS OpenBLAS raises spurious warnings on matmul for denormal inputs.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)


def main():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    features = meta["features"]

    frame = pd.read_csv(TEST_CSV)
    X = joblib.load(MODEL_DIR / "scaler.joblib").transform(frame[features])
    y = frame[TARGET].to_numpy()

    models = {key: joblib.load(MODEL_DIR / f"{key}.joblib") for key in MODEL_KEYS}
    probability = {key: model.predict_proba(X)[:, 1] for key, model in models.items()}
    prediction = {key: model.predict(X) for key, model in models.items()}

    rng = np.random.default_rng(SEED)
    n = len(y)
    resamples = [rng.integers(0, n, n) for _ in range(N_BOOTSTRAP)]

    intervals = {}
    print(f"Bootstrap 95% intervals ({N_BOOTSTRAP} resamples)\n")
    print(f"  {'model':22} {'AUC':>18} {'MCC':>20}")
    for key in MODEL_KEYS:
        aucs, mccs = [], []
        for index in resamples:
            if y[index].min() == y[index].max():
                continue  # a resample with one class has no defined AUC
            aucs.append(roc_auc_score(y[index], probability[key][index]))
            mccs.append(matthews_corrcoef(y[index], prediction[key][index]))
        intervals[key] = {
            "auc": {
                "point": float(roc_auc_score(y, probability[key])),
                "low": float(np.percentile(aucs, 2.5)),
                "high": float(np.percentile(aucs, 97.5)),
            },
            "mcc": {
                "point": float(matthews_corrcoef(y, prediction[key])),
                "low": float(np.percentile(mccs, 2.5)),
                "high": float(np.percentile(mccs, 97.5)),
            },
        }
        a, m = intervals[key]["auc"], intervals[key]["mcc"]
        print(
            f"  {key:22} {a['point']:.4f} [{a['low']:.4f}, {a['high']:.4f}]"
            f"   {m['point']:.4f} [{m['low']:.4f}, {m['high']:.4f}]"
        )

    print("\nDecision-tree threshold sweep\n")
    print(f"  {'cut':>5} {'flagged':>8} {'precision':>10} {'recall':>8} {'f1':>8} {'mcc':>8}")
    tree_scores = probability["decision_tree"]
    sweep = []
    for threshold in (0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
        labels = (tree_scores >= threshold).astype(int)
        row = {
            "threshold": threshold,
            "flagged": int(labels.sum()),
            "precision": float(precision_score(y, labels, zero_division=0)),
            "recall": float(recall_score(y, labels, zero_division=0)),
            "f1": float(f1_score(y, labels, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, labels)),
        }
        sweep.append(row)
        print(
            f"  {threshold:>5.2f} {row['flagged']:>8,} {row['precision']:>10.4f}"
            f" {row['recall']:>8.4f} {row['f1']:>8.4f} {row['mcc']:>8.4f}"
        )

    tree = models["decision_tree"]
    importances = sorted(
        zip(features, map(float, tree.feature_importances_)), key=lambda pair: -pair[1]
    )
    print("\nDecision-tree feature importances\n")
    for name, value in importances[:8]:
        print(f"  {name:12} {value:.4f}")
    print(f"  leaves: {tree.n_leaves_}  depth <= {tree.max_depth}")

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "analysis.json").write_text(
        json.dumps(
            {
                "bootstrap_resamples": N_BOOTSTRAP,
                "confidence_intervals": intervals,
                "decision_tree_threshold_sweep": sweep,
                "decision_tree_importances": importances,
                "decision_tree_leaves": tree.n_leaves_,
            },
            indent=2,
        )
    )
    print(f"\nwrote {REPORT_DIR / 'analysis.json'}")


if __name__ == "__main__":
    main()
