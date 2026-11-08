# Credit Card Default Prediction

An end-to-end machine learning project that predicts whether a credit card client will default on their next month's payment. Five classification algorithms are trained on the same dataset, compared across six evaluation metrics, and served through an interactive Streamlit application.

**All five algorithms are implemented from scratch** in the [`models/`](models/) package — one file per algorithm, each inheriting from a common `Model` base class. scikit-learn is used only for the train/test split, the scaler and the metric functions, and a validation harness checks each hand-written implementation against scikit-learn's reference version on identical data.

**Machine Learning — Assignment 2 (M.Tech AIML, BITS Pilani WILP)**

---

## a. Problem statement

Credit card issuers carry risk on every account. If the bank can identify likely defaulters *before* the due date, it can lower credit limits, offer restructuring, or provision capital more accurately. Missing a defaulter costs the full outstanding balance; wrongly flagging a good customer costs a relationship.

This is a **binary classification** problem: given a client's credit limit, demographics, six months of repayment history, bill statements and payment amounts, predict whether they will default next month (`1`) or not (`0`).

The dataset is imbalanced — only **22.1%** of clients default — so accuracy alone is misleading. A model that predicts "no default" for every client already scores **0.779**. This is why MCC, AUC and recall are reported alongside accuracy.

---

## b. Dataset description

**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients (ID 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
Originally published by Yeh & Lien (2009). Data from 30,000 clients of a Taiwanese bank, April–September 2005.

| Property | Value |
|---|---|
| Raw instances | 30,000 |
| Instances after cleaning | **29,965** (35 exact duplicates removed) |
| Raw columns | 25 (1 ID + 23 features + 1 target) |
| Features used | **23** |
| Target | `default` — 1 = defaulted next month |
| Class balance | 23,364 non-default (77.9%) / 6,636 default (22.1%) |
| Missing values | None |
| Feature types | All numeric (demographics are integer-coded) |

### Feature groups

| Group | Columns | Description |
|---|---|---|
| Credit & demographics | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Amount of credit granted (NT$), sex, education level, marital status, age in years |
| Repayment status (6 months) | `PAY_0`, `PAY_2` … `PAY_6` | Months of delay: `-2` no consumption, `-1` paid in full, `0` revolving credit, `1`–`8` months delinquent |
| Bill statement (6 months) | `BILL_AMT1` … `BILL_AMT6` | Amount billed each month. **Negative values are legitimate credit balances**, not errors |
| Previous payment (6 months) | `PAY_AMT1` … `PAY_AMT6` | Amount actually paid in each month |

### Cleaning performed (`prepare_data.py`)

1. Dropped the `ID` column — it is a row index with no predictive value.
2. Renamed `default payment next month` → `default`.
3. `EDUCATION` contained undocumented codes `0`, `5` and `6` (345 rows total, not described in the UCI documentation) — folded into a single `4 (others)` category.
4. `MARRIAGE` contained an undocumented code `0` (54 rows) — folded into `3 (others)`.
5. Removed 35 exact duplicate rows.

### Train / test split

80 / 20 stratified split (`random_state=42`) → **23,972 train** / **5,993 test** rows. The test partition is committed as `test_data.csv` and is the file used by the Streamlit app.

**Preprocessing:** a `StandardScaler` is fitted on the training partition only and applied to both partitions. It is persisted to `model/scaler.joblib` so the app applies the *identical* transformation — critical, because `LIMIT_BAL` (10k–1M) and `AGE` (21–79) differ by five orders of magnitude, which would otherwise dominate distance-based and margin-based models.

---

## c. GitHub repository link

`https://github.com/M0706/credit-card-default-classifier`

---

## d. Models used and results

All five models were trained on the same 23 features and evaluated on the same 5,993 held-out rows. `Decision Tree` depth and `KNN` `k` were chosen on a validation split carved out of the training data, and the selected configuration was then **refit on the full 23,972-row training set** before scoring.

| Model | File | Training method |
|---|---|---|
| Logistic Regression | [`models/logistic.py`](models/logistic.py) | L2-penalised log loss minimised by Newton's method (IRLS) — 7 iterations |
| Decision Tree | [`models/tree.py`](models/tree.py) | CART: greedy splits on Gini impurity, quantile candidate thresholds |
| K-Nearest Neighbours | [`models/knn.py`](models/knn.py) | Memorises the training set; distance-weighted vote over 15 neighbours |
| Naive Bayes | [`models/naive_bayes.py`](models/naive_bayes.py) | Gaussian likelihoods with the conditional-independence assumption |
| Support Vector Machine | [`models/svm.py`](models/svm.py) | RBF kernel, soft margin, solved by simplified SMO (Platt) |

`validate_models.py` fits each hand-written model and scikit-learn's version on the identical split and compares them. The largest AUC difference across all five is **0.0035** (the decision tree, where quantile thresholds replace sklearn's exhaustive search).

All five share one interface, defined by `models/base.py`, so training and serving code never branch on the algorithm:

```python
model.fit(X, y)              # learn
model.predict_proba(X)       # -> (n_samples, 2) probabilities
model.predict(X)             # -> 0/1 labels at a 0.5 cut-off
model.save(path)             # joblib, same artefact the app loads
```

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8133 | 0.7160 | **0.7285** | 0.2489 | 0.3710 | 0.3494 |
| Decision Tree | **0.8195** | **0.7472** | 0.6658 | 0.3695 | 0.4753 | **0.4007** |
| KNN | 0.8066 | 0.7333 | 0.6152 | 0.3363 | 0.4349 | 0.3520 |
| Naive Bayes | 0.6915 | 0.7283 | 0.3864 | **0.6704** | **0.4902** | 0.3140 |
| Support Vector Machine | 0.8133 | 0.7105 | 0.6448 | 0.3477 | 0.4517 | 0.3755 |

**Statistical caveat.** These are point estimates. Bootstrap 95% intervals (300 resamples of the test set) overlap for the Decision Tree against KNN, Naive Bayes and SVM on MCC, so most pairwise gaps are not statistically defensible. Only two claims survive the intervals: the Decision Tree's MCC advantage over Naive Bayes (**0.364–0.428** vs 0.288–0.338) and its AUC advantage over the SVM (**0.733–0.763** vs 0.695–0.727). Full intervals are in [`docs/05_why_these_results.md`](docs/05_why_these_results.md).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (**0.7285**) but the lowest recall of the four linear-family models (0.2489) — it flags a default only when very confident, so it misses three out of every four actual defaulters. Because it minimises log-loss under a 78/22 imbalance, Newton's method pushes the weights toward the majority class: the intercept settles at −1.469, close to the log-odds of the base rate. Good AUC (0.7160) relative to that poor recall shows its *ranking* is reasonable even though the 0.5 threshold is far too conservative. It also cannot express the interaction effects that matter here — a high bill coupled with a rising delinquency count is far riskier than either alone. |
| Decision Tree | Best on accuracy (**0.8195**), AUC (**0.7472**) and MCC (**0.4007**) — the only model to lead on all three. Tuning chose `max_depth=6` over depth 4 on validation F1 (0.4609 vs 0.4605 — a tie in practice) and over an unrestricted tree (0.4323), confirming that unbounded depth memorises noise. The fitted tree has 44 leaves, and one variable dominates: `PAY_0` (most recent repayment status) accounts for **66.7%** of impurity reduction, with `PAY_2` adding 13.8%. Fully interpretable: any path can be shown to a loan officer and defended. |
| KNN | Competitive AUC (0.7333) but the weakest operational profile. The saved model stores all 23,972 training points (4.6 MB) and must compare every new client against each of them, so latency grows linearly with the dataset. With 23 features it also suffers from the curse of dimensionality — distances concentrate and neighbours stop being meaningfully "similar". `k=15` with distance weighting won the validation sweep (F1 0.4365 vs 0.4218 at k=25 and 0.3940 at k=75); larger neighbourhoods increasingly blurred the minority class until it disappeared. |
| Naive Bayes | The mirror image of Logistic Regression: highest recall (0.6704) and highest F1 (0.4902) but lowest accuracy (0.6915), precision (0.3864) and MCC (0.3140). Gaussian NB assumes each feature is conditionally independent and normally distributed. Both assumptions fail badly here — `BILL_AMT1`…`BILL_AMT6` are near-duplicates, and the bill/payment distributions are heavily skewed. The redundant evidence is effectively counted six times, pushing predicted probabilities toward the minority class. It catches two-thirds of defaulters but raises four false alarms for every six true ones. |
| Support Vector Machine | Accuracy 0.8133 with MCC 0.3755 — second best on MCC, but it pays for that in AUC (0.7105, lowest of the five). It uses an RBF kernel with `C=1.0` and `gamma='scale'` on standardised features, and finds a genuinely non-linear boundary that captures the delinquency-by-bill interaction Logistic Regression misses. **1,877 of the 4,000 subsampled training points became support vectors** — the data is noisy enough that almost half the points sit on or inside the margin. The precomputed kernel matrix is why training uses a stratified 4,000-row subsample instead of all 23,972 rows; `validate_models.py` confirms the SMO implementation reproduces scikit-learn exactly on that same subsample. Probabilities are a monotone sigmoid of the decision value, so the *ranking* (and therefore AUC) is exact but the probabilities are not calibrated. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It wins on MCC (0.4007) — the most trustworthy metric under class imbalance — and on AUC (0.7472) and accuracy (0.8195), while staying competitive on F1. It additionally offers full interpretability and microsecond inference from a 5 KB artefact. If the business objective were "catch as many defaulters as possible at any cost", Naive Bayes would win on recall (0.6704) — but its MCC shows it is mostly guessing the minority class. The real conclusion is that **four of the five sit only 2.8–4.1 points above the 0.7787 majority-class baseline on accuracy, and Naive Bayes sits 8.7 points below it** — so the choice between them should be driven by MCC and the precision/recall trade-off, not accuracy. |

---

## Project structure

```
.
├── app.py                  # Streamlit application
├── prepare_data.py         # Raw .xls -> cleaned CSV
├── train_models.py         # Trains, evaluates and saves all five models
├── validate_models.py      # Checks each from-scratch model against scikit-learn
├── analysis.py             # Bootstrap intervals, threshold sweep, importances
├── requirements.txt
├── README.md
├── test_data.csv           # 5,993 held-out rows (features + target)
├── models/                 # Hand-written implementations
│   ├── base.py             # Model ABC: fit / predict_proba / predict / save / load
│   ├── logistic.py         # LogisticRegressionModel — Newton (IRLS)
│   ├── tree.py             # DecisionTreeModel — CART, Gini
│   ├── knn.py              # KNNModel — distance-weighted voting
│   ├── naive_bayes.py      # GaussianNBModel — Gaussian likelihoods
│   └── svm.py              # SVMModel — RBF kernel + SMO
├── data/
│   ├── raw/                # Original UCI .xls download
│   └── credit_default.csv  # Cleaned dataset (29,965 rows)
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── scaler.joblib        # Fitted StandardScaler — required for inference
│   └── metadata.json        # Feature order, target name, positive class
└── reports/
    ├── metrics.json         # Metric values for all five models
    ├── details.json         # Confusion matrices and classification reports
    ├── analysis.json        # Bootstrap intervals, threshold sweep, importances
    ├── validation.json      # From-scratch vs scikit-learn comparison
    └── training_config.json # Chosen hyperparameters and split sizes
```

## Study notes

`docs/` explains the project from first principles — theory, metrics arithmetic, data
pipeline, code walkthrough and viva preparation. Every number quoted there is produced by
this repository's code.

| Document | Contents |
|---|---|
| [`docs/01_theory.md`](docs/01_theory.md) | How each of the five algorithms works, with the maths |
| [`docs/02_metrics.md`](docs/02_metrics.md) | All six metrics, with worked arithmetic from our confusion matrices |
| [`docs/03_data_pipeline.md`](docs/03_data_pipeline.md) | Raw `.xls` → cleaned CSV → scaled matrices, and the leakage traps |
| [`docs/04_code_walkthrough.md`](docs/04_code_walkthrough.md) | Line-by-line tour of every script, including the `models/` package |
| [`docs/05_why_these_results.md`](docs/05_why_these_results.md) | Mechanism behind every reported number, plus bootstrap intervals |
| [`docs/06_viva_prep.md`](docs/06_viva_prep.md) | 41 defence questions with answers |
| [`docs/07_implementations.md`](docs/07_implementations.md) | The five algorithms from scratch: the maths in each file, the bugs to avoid, and the proof they match scikit-learn |

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To regenerate the artefacts from scratch:

```bash
python prepare_data.py     # writes data/credit_default.csv
python train_models.py     # writes model/*.joblib, reports/*, test_data.csv
python validate_models.py  # from-scratch vs scikit-learn, writes reports/validation.json
python analysis.py         # bootstrap intervals + threshold sweep
```

## Streamlit application

**Live app:** `<paste your Streamlit Community Cloud URL here>`

The app provides:

1. **CSV upload** — evaluate any dataset with the 23 feature columns plus the `default` target. A bundled `test_data.csv` is used if nothing is uploaded.
2. **Model selection dropdown** — switch between all five pre-trained classifiers.
3. **Evaluation metrics** — accuracy, AUC, precision, recall, F1 and MCC for the uploaded data.
4. **Confusion matrix and classification report** — per-class precision/recall/F1/support.

The app loads the persisted `.joblib` artefacts and only calls `predict` / `predict_proba`. **No model is retrained on the uploaded data**; the saved `StandardScaler` is reused so uploaded rows are transformed exactly as the training data was.

## Evaluation metric definitions

| Metric | Definition | Why it matters here |
|---|---|---|
| Accuracy | (TP + TN) / total | Misleading at 22% positive rate — 0.779 is achievable by predicting all-negative |
| AUC | Area under the ROC curve | Threshold-independent ranking quality |
| Precision | TP / (TP + FP) | Of clients flagged as defaulters, how many actually default |
| Recall | TP / (TP + FN) | Of actual defaulters, how many were caught — the metric the bank cares about most |
| F1 | Harmonic mean of precision and recall | Single-number balance of the two |
| MCC | Correlation between prediction and truth, range −1…+1 | Reliable under class imbalance; only high when **both** classes are predicted well |
