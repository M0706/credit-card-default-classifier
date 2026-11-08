# Credit Card Default Prediction

Predicts whether a credit card client will default on their next payment, using the UCI
Default of Credit Card Clients dataset. Five classifiers are compared on six metrics, and
the trained models are served through a Streamlit app.

All five are written by hand in `models/`, one file per algorithm, with a shared base class.
NumPy is the only numerical dependency. scikit-learn handles the train/test split, the
scaler and the metric functions, and `validate_models.py` uses its implementations as a
reference to check mine against.

Machine Learning Assignment 2, M.Tech AIML, BITS Pilani WILP.

---

## a. Problem statement

A bank loses the outstanding balance when a client defaults, and loses goodwill when it
flags someone who would have paid. If defaults could be spotted even a month ahead, there
is room to act: lower the limit, restructure the payment, set aside capital.

This is a binary classification problem. Each client is described by 23 numeric columns
(credit limit, age, sex, education, marital status, six months of repayment status, six
bill amounts and six payment amounts), and the target is 1 if the client defaults next
month, 0 otherwise.

The positive class is 22.1% of the data. A model that answers "no default" every time is
already right 77.9% of the time, so accuracy on its own says very little here. MCC, AUC
and recall carry most of the argument.

---

## b. Dataset description

**Source:** [UCI Machine Learning Repository, Default of Credit Card Clients (ID 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
Published by Yeh & Lien (2009). The records come from 30,000 clients of a Taiwanese bank
between April and September 2005.

| Property | Value |
|---|---|
| Raw instances | 30,000 |
| Instances after cleaning | **29,965** (35 exact duplicates removed) |
| Raw columns | 25 (1 ID + 23 features + 1 target) |
| Features used | **23** |
| Target | `default`, 1 = defaulted next month |
| Class balance | 23,364 non-default (77.9%) / 6,636 default (22.1%) |
| Missing values | None |
| Feature types | All numeric (demographics are integer-coded) |

### What the columns are

| Group | Columns | Description |
|---|---|---|
| Credit and demographics | `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Credit granted (NT$), sex, education level, marital status, age in years |
| Repayment status, 6 months | `PAY_0`, `PAY_2` … `PAY_6` | Months of delay. `-2` no consumption, `-1` paid in full, `0` revolving credit, `1`–`8` months delinquent |
| Bill statement, 6 months | `BILL_AMT1` … `BILL_AMT6` | Amount billed each month. Negative values are credit balances, not errors |
| Previous payment, 6 months | `PAY_AMT1` … `PAY_AMT6` | Amount actually paid each month |

### Cleaning (`prepare_data.py`)

1. Dropped the `ID` column, which is just a row index.
2. Renamed `default payment next month` to `default`.
3. `EDUCATION` contains codes `0`, `5` and `6` that the UCI documentation does not
   describe (345 rows). I folded them into a single `4 (others)` category.
4. `MARRIAGE` has an undocumented code `0` (54 rows), folded into `3 (others)`.
5. Removed 35 exact duplicate rows.

### Train / test split

80/20, stratified, `random_state=42`. That gives **23,972 training rows** and **5,993
test rows**. The test partition is committed as `test_data.csv` and is the file the
Streamlit app uses by default.

The `StandardScaler` is fitted on the training partition only and saved to
`model/scaler.joblib`, so the app applies the same transformation at serving time. Scaling
matters most for KNN and SVM. `LIMIT_BAL` runs into the hundreds of thousands and `AGE`
into the tens, and without scaling the distance between two clients would be almost
entirely credit limit.

---

## c. GitHub repository link

`https://github.com/M0706/credit-card-default-classifier`

---

## d. Models used and results

All five models see the same 23 features and are scored on the same 5,993 held-out rows.
Decision tree depth and KNN `k` were chosen on a validation split taken out of the
training data, and the winning settings were then refit on all 23,972 rows before scoring.

| Model | File | Training method |
|---|---|---|
| Logistic Regression | [`models/logistic.py`](models/logistic.py) | L2-penalised log loss, minimised with Newton's method (IRLS), 7 iterations |
| Decision Tree | [`models/tree.py`](models/tree.py) | CART, greedy splits on Gini impurity, quantile candidate thresholds |
| K-Nearest Neighbours | [`models/knn.py`](models/knn.py) | Stores the training set, distance-weighted vote over 15 neighbours |
| Naive Bayes | [`models/naive_bayes.py`](models/naive_bayes.py) | Gaussian likelihoods, conditional independence assumed |
| Support Vector Machine | [`models/svm.py`](models/svm.py) | RBF kernel, soft margin, solved by simplified SMO |

`validate_models.py` fits each hand-written model next to its scikit-learn equivalent on
the same split. The largest AUC gap is **0.0035**, on the decision tree, where I use
quantile cut points instead of testing every midpoint.

The five classes share one interface from `models/base.py`, so the training and serving
code never checks which algorithm it is working with:

```python
model.fit(X, y)              # learn
model.predict_proba(X)       # -> (n_samples, 2) probabilities
model.predict(X)             # -> 0/1 labels at a 0.5 cut-off
model.save(path)             # joblib, the same file the app loads
```

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8133 | 0.7160 | **0.7285** | 0.2489 | 0.3710 | 0.3494 |
| Decision Tree | **0.8195** | **0.7472** | 0.6658 | 0.3695 | 0.4753 | **0.4007** |
| KNN | 0.8066 | 0.7333 | 0.6152 | 0.3363 | 0.4349 | 0.3520 |
| Naive Bayes | 0.6915 | 0.7283 | 0.3864 | **0.6704** | **0.4902** | 0.3140 |
| Support Vector Machine | 0.8133 | 0.7105 | 0.6448 | 0.3477 | 0.4517 | 0.3755 |

These are point estimates on 5,993 rows, so they carry sampling noise. Bootstrapping the
test set 300 times gives the tree an AUC interval of 0.733 to 0.763 and an MCC interval of
0.364 to 0.428; Naive Bayes' MCC interval is 0.288 to 0.338. Most of the other pairwise
gaps overlap, so I would only claim two rankings with any confidence: the tree beats the
SVM on AUC, and it beats Naive Bayes on MCC. All five intervals are in
[`docs/05_why_these_results.md`](docs/05_why_these_results.md).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision of the five (0.7285) and the lowest recall of the four non-Naive-Bayes models (0.2489). At a 0.5 cut-off it flags a client only when it is very confident, so it catches about one defaulter in four. The AUC of 0.7160 is much better than the F1 suggests: the ranking is reasonable and the threshold is what costs it. It also cannot express the interaction between a large bill and rising delinquency, which is where a good part of the signal sits. The intercept lands at -1.4687, close to the log-odds of the base rate. |
| Decision Tree | Best accuracy (0.8195), AUC (0.7472) and MCC (0.4007). Depth 6 came out of validation, but the margin over depth 4 is tiny (F1 0.4609 against 0.4605), and depth 8 and above does worse (0.4432, then 0.4323). The fitted tree has 44 leaves, and `PAY_0` accounts for 66.7% of the impurity reduction, with `PAY_2` adding 13.8%. Any path through it can be read as a rule and shown to a loan officer. |
| KNN | AUC 0.7333, close to the tree, but the most expensive model to serve: the saved file is 4.6 MB because it keeps all 23,972 training rows, and every prediction compares the new client against each of them. `k=15` with distance weighting scored 0.4365 validation F1, against 0.4218 at `k=25` and 0.3940 at `k=75`. Larger neighbourhoods wash out the minority class. With 23 features the distances also concentrate, so neighbours stop being meaningfully similar. |
| Naive Bayes | Best recall (0.6704) and best F1 (0.4902), but worst accuracy (0.6915, below the 0.7787 baseline), precision (0.3864) and MCC (0.3140). It flagged 2,301 clients, about 38% of the test set against a true rate of 22%. The six `BILL_AMT` columns are nearly the same measurement, and treating them as independent evidence counts the same fact six times. The model ends up pushed toward the minority class: 889 defaulters found, 1,412 false alarms. |
| Support Vector Machine | Accuracy 0.8133 and MCC 0.3755, second best on MCC, but the weakest AUC (0.7105). RBF kernel with `C=1.0` and `gamma='scale'`, left untuned. 1,877 of the 4,000 training points became support vectors, which says most of the data sits near the boundary. Training uses a stratified 4,000-row subsample because the precomputed kernel matrix grows with n²; all 23,972 rows would need about 4.6 GB. Probabilities are a sigmoid of the decision value, fine for ranking but not calibrated. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It leads on MCC, the metric I trust most under this class balance, and also on AUC and accuracy, and the artefact is 5 KB and answers in microseconds. Naive Bayes wins on recall, but its MCC says that recall mostly comes from predicting default far too often. Worth keeping in mind: four of the five sit only 2.8 to 4.1 points above the majority-class baseline and Naive Bayes sits 8.7 points below it, so accuracy alone would have ranked these models badly. |

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
│   ├── logistic.py         # LogisticRegressionModel, Newton (IRLS)
│   ├── tree.py             # DecisionTreeModel, CART with Gini
│   ├── knn.py              # KNNModel, distance-weighted voting
│   ├── naive_bayes.py      # GaussianNBModel, Gaussian likelihoods
│   └── svm.py              # SVMModel, RBF kernel with SMO
├── data/
│   ├── raw/                # Original UCI .xls download
│   └── credit_default.csv  # Cleaned dataset (29,965 rows)
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── scaler.joblib       # Fitted StandardScaler, needed at inference
│   └── metadata.json       # Feature order, target name, positive class
└── reports/
    ├── metrics.json        # Metric values for all five models
    ├── details.json        # Confusion matrices and classification reports
    ├── analysis.json       # Bootstrap intervals, threshold sweep, importances
    ├── validation.json     # From-scratch vs scikit-learn comparison
    └── training_config.json
```

## Study notes

`docs/` explains the project from the ground up: the theory behind each algorithm, the
arithmetic behind each metric, the data pipeline, a walkthrough of the code, and answers to
the questions a viva is likely to ask. Every number in those documents comes from this
repository's own output.

| Document | Contents |
|---|---|
| [`docs/01_theory.md`](docs/01_theory.md) | How each of the five algorithms works |
| [`docs/02_metrics.md`](docs/02_metrics.md) | The six metrics, worked through on our confusion matrices |
| [`docs/03_data_pipeline.md`](docs/03_data_pipeline.md) | Raw `.xls` to cleaned CSV to scaled matrices, and the leakage traps |
| [`docs/04_code_walkthrough.md`](docs/04_code_walkthrough.md) | A tour of every script, including the `models/` package |
| [`docs/05_why_these_results.md`](docs/05_why_these_results.md) | Why each model produced its numbers, with the bootstrap intervals |
| [`docs/06_viva_prep.md`](docs/06_viva_prep.md) | 41 defence questions and answers |
| [`docs/07_implementations.md`](docs/07_implementations.md) | The five algorithms from scratch, and how they were checked |

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

`requirements.txt` uses minimum versions rather than exact pins, so the install works on
any Python 3.9 or later. The saved models were trained with scikit-learn 1.6.1 and load
under newer releases.

To rebuild everything from the raw data:

```bash
python prepare_data.py     # writes data/credit_default.csv
python train_models.py     # writes model/*.joblib, reports/*, test_data.csv
python validate_models.py  # from-scratch vs scikit-learn
python analysis.py         # bootstrap intervals and threshold sweep
```

## Streamlit application

**Live app:** https://credit-card-classifier.streamlit.app/

The app has four parts:

1. **CSV upload.** Any file with the 23 feature columns plus a `default` column. If you
   upload nothing, it falls back to the bundled `test_data.csv`.
2. **Model dropdown.** Switches between the five saved classifiers.
3. **Metrics.** Accuracy, AUC, precision, recall, F1 and MCC on the uploaded rows.
4. **Confusion matrix and classification report**, with per-class precision, recall, F1
   and support.

The app loads the `.joblib` files and calls `predict` and `predict_proba`. Nothing is
retrained on the uploaded data, and the saved scaler is reused so the rows are transformed
exactly as the training data was.

## Evaluation metric definitions

| Metric | Definition | Why it matters here |
|---|---|---|
| Accuracy | (TP + TN) / total | Misleading at a 22% positive rate, since 0.779 is available by predicting all-negative |
| AUC | Area under the ROC curve | Ranking quality, independent of the threshold |
| Precision | TP / (TP + FP) | Of the clients flagged as defaulters, how many actually default |
| Recall | TP / (TP + FN) | Of the actual defaulters, how many were caught |
| F1 | Harmonic mean of precision and recall | One number balancing the two |
| MCC | Correlation between prediction and truth, from -1 to +1 | Stays reliable under imbalance, and only scores high when both classes are handled well |
