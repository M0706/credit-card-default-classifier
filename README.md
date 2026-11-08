# Credit Card Default Prediction

An end-to-end machine learning project that predicts whether a credit card client will default on their next month's payment. Five classification algorithms are trained on the same dataset, compared across six evaluation metrics, and served through an interactive Streamlit application.

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

`https://github.com/<your-username>/credit-card-default-classifier`

<!-- TODO: replace <your-username> after creating the repository -->

---

## d. Models used and results

All five models were trained on the same 23 features and evaluated on the same 5,993 held-out rows. `Decision Tree` depth and `KNN` `k` were chosen on a validation split carved out of the training data, and the selected configuration was then **refit on the full 23,972-row training set** before scoring; all other hyperparameters are library defaults on standardised features.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8133 | 0.7160 | 0.7275 | 0.2496 | 0.3717 | 0.3496 |
| Decision Tree | **0.8163** | **0.7341** | 0.6490 | 0.3695 | 0.4709 | **0.3912** |
| KNN | 0.8073 | 0.7327 | 0.6179 | 0.3379 | 0.4369 | 0.3545 |
| Naive Bayes | 0.6915 | 0.7283 | 0.3864 | **0.6704** | **0.4902** | 0.3140 |
| Support Vector Machine | 0.8161 | 0.7173 | 0.6713 | 0.3311 | 0.4434 | 0.3794 |

**Statistical caveat.** These are point estimates. Bootstrap 95% intervals (300 resamples of the test set) overlap across all five models on AUC, and overlap for the Decision Tree versus SVM and KNN on MCC — so the only difference that is statistically defensible is the Decision Tree's MCC advantage over Naive Bayes (0.360–0.418 vs 0.287–0.338). Full intervals are in [`docs/05_why_these_results.md`](docs/05_why_these_results.md).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.7275) but the lowest recall of the four linear-family models (0.2496) — it flags a default only when very confident, so it misses roughly 3 out of every 4 actual defaulters. Because it optimises log-loss under a 78/22 imbalance, it is pushed toward the majority class. Good AUC (0.7160) relative to its poor recall shows its *ranking* is reasonable even though the 0.5 threshold is far too conservative. It also cannot express the interaction effects that matter here — a high bill coupled with a rising delinquency count is far riskier than either alone. |
| Decision Tree | Best on accuracy (0.8163), AUC (0.7341) **and** MCC (0.3912) — the only model to lead on all three. Tuning selected a shallow tree (`max_depth=4`, `min_samples_leaf=50`, 15 leaves): the fully-grown tree scored a *lower* validation F1 (0.4290 vs 0.4628), confirming that unrestricted depth memorises noise. Only a handful of axis-aligned splits are needed because one variable dominates — `PAY_0` (most recent repayment status) accounts for **73.5%** of the tree's impurity reduction, with `PAY_2` adding 14.4%. Fully interpretable: the decision path can be shown to a loan officer and defended. |
| KNN | Competitive AUC (0.7327) but the weakest operational profile: the saved model is 3.5 MB and must compare every new client against all 23,972 training points, so latency grows linearly with the dataset. With 23 features it also suffers from the curse of dimensionality — distances concentrate and neighbours stop being meaningfully "similar". Distance weighting with `k=15` was selected over larger neighbourhoods, which increasingly blurred the minority class. Notably, refitting on the full training set slightly *lowered* its MCC (0.3611 → 0.3545), because a larger reference set changes which 15 neighbours vote near the decision boundary. |
| Naive Bayes | The mirror image of Logistic Regression: highest recall (0.6704) and highest F1 (0.4902) but lowest accuracy (0.6915), precision (0.3864) and MCC (0.3140). Gaussian NB assumes each feature is conditionally independent and normally distributed. Both assumptions fail badly here — `BILL_AMT1`…`BILL_AMT6` are near-duplicates of one another, and the bill/payment distributions are heavily skewed. The redundant evidence is effectively counted six times, pushing predicted probabilities toward the minority class. It catches most defaulters but raises four false alarms for every six true ones, and its probability estimates should not be used for pricing decisions. |
| Support Vector Machine | Accuracy 0.8161 — effectively level with the Decision Tree — with MCC 0.3794, third best. Uses an RBF kernel with `C=1.0`, `gamma='scale'` on standardised features, and it finds a non-linear boundary that captures the delinquency-by-bill interaction Logistic Regression misses. Its recall (0.3311) is still timid. Only **10,506 of 23,972** training points became support vectors, but predicting requires evaluating the kernel against every one of them, hence the 2.0 MB artefact and the highest inference cost of the five. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It wins on MCC (0.3912) — the most trustworthy metric under class imbalance — and on AUC (0.7341) and accuracy (0.8163), while staying competitive on F1. It additionally offers full interpretability and millisecond inference from a 3 KB artefact. If the business objective were "catch as many defaulters as possible at any cost", Naive Bayes would win on recall (0.6704) — but its MCC shows it is mostly guessing the minority class. The real conclusion is that **all five models sit only ~3 points above the 0.779 majority-class baseline on accuracy**, so the choice between them should be driven by MCC and the precision/recall trade-off, not accuracy. |

---

## Project structure

```
.
├── app.py                  # Streamlit application
├── prepare_data.py         # Raw .xls -> cleaned CSV
├── train_models.py         # Trains, evaluates and saves all five models
├── requirements.txt
├── README.md
├── test_data.csv           # 5,993 held-out rows (features + target)
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
    └── details.json         # Confusion matrices and classification reports
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
| [`docs/04_code_walkthrough.md`](docs/04_code_walkthrough.md) | Line-by-line tour of all three scripts |
| [`docs/05_why_these_results.md`](docs/05_why_these_results.md) | Mechanism behind every reported number, plus bootstrap intervals |
| [`docs/06_viva_prep.md`](docs/06_viva_prep.md) | 40 defence questions with answers |

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
