# 3. The Data Pipeline — Raw Spreadsheet to Model-Ready Matrix

## 3.1 The raw file

`data/raw/default of credit card clients.xls` is a legacy Excel file. It has a trap:

```
Row 0:  X1, X2, X3, ... , default payment next month,      ← metadata / dummy header
Row 1:  ID, LIMIT_BAL, SEX, ..., default payment next month ← the REAL header
Row 2+: actual client records
```

So the code reads it with **`header=1`**:

```python
df = pd.read_excel(RAW, header=1)
```

Using the default `header=0` yields columns named `X1…X23` and one junk row. If you ever
see "`X6`" in your feature list, this is why.

Reading `.xls` (as opposed to `.xlsx`) requires the `xlrd` package — the reason it
appears in `requirements.txt`.

## 3.2 What the 25 columns actually mean

| Column(s) | Type | Meaning | Gotcha |
|---|---|---|---|
| `ID` | int | Row index 1…30000 | **No predictive value — dropped** |
| `LIMIT_BAL` | int | Credit limit in NT$ (New Taiwan dollars) | Range 10,000–1,000,000 |
| `SEX` | int | 1 = male, 2 = female | Already integer-coded |
| `EDUCATION` | int | 1 = graduate, 2 = university, 3 = high school, 4 = others | Undocumented codes 0, 5, 6 |
| `MARRIAGE` | int | 1 = married, 2 = single, 3 = others | Undocumented code 0 |
| `AGE` | int | Years | 21–79 |
| `PAY_0` … `PAY_6` | int | Repayment status, most recent … 6 months ago | **Negative values are codes, not errors** |
| `BILL_AMT1` … `BILL_AMT6` | int | Amount billed each month | **Negative = credit balance** |
| `PAY_AMT1` … `PAY_AMT6` | int | Amount actually paid each month | Never negative |
| `default payment next month` | int | 0 / 1 target | Renamed to `default` |

### Decoding `PAY_*`

These are the most important columns in the dataset, and the values are not magnitudes:

| Value | Meaning |
|---|---|
| −2 | No consumption that month (no card activity) |
| −1 | Paid in full |
| 0 | Revolving credit (used some credit, paid the minimum) |
| 1 | Payment delayed 1 month |
| 2–8 | Payment delayed 2–8 months |

So `PAY_0 = -2` is *better* than `PAY_0 = -1`, which is better than `0`, which is better
than `1`. Read as a plain number the ordering happens to hold from −1 upward, but the
gap between −2 and −1 does **not** mean "one unit of lateness". Treating these as
continuous is a simplification — a strictly correct approach would one-hot encode them
as 11 ordered categories. We kept them numeric because tree-based and distance-based
models handle the ordering well, and it keeps the comparison across five models clean.
This is a fair thing to raise as a limitation.

### Negative `BILL_AMT`

A negative bill is not corrupt data — it is a **credit balance** (the client overpaid or
received a refund). `BILL_AMT6` goes as low as −339,603. Deleting these rows would
silently discard the most creditworthy customers in the dataset.

## 3.3 Cleaning steps, and why each one

```python
df = df.drop(columns=["ID"])
df = df.rename(columns={"default payment next month": TARGET})

df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})

df = df.drop_duplicates().reset_index(drop=True)
```

### Why fold the undocumented codes

The UCI documentation defines `EDUCATION ∈ {1,2,3,4}`. The data contains:

| Code | Count | Documented? |
|---|---|---|
| 0 | 14 | ❌ no |
| 5 | 280 | ❌ no |
| 6 | 51 | ❌ no |

345 rows carry values the dataset's own authors never defined. Leaving them creates
three categories with 14, 280 and 51 examples — too few to learn anything from, and
unexplainable in a report. Folding them into the existing `4 = others` bucket is the
standard treatment. Same logic for `MARRIAGE = 0` (54 rows).

**The alternative** is to drop those 345 + 54 rows. We keep them because they still carry
valid credit-limit, bill and repayment information — only one demographic field is
unknown.

### Why drop duplicates

There were 35 exact duplicate rows. A duplicated row gets effectively double weight in
training and, worse, if a duplicate straddles the train/test boundary the model has
literally seen that exact client in training — a subtle leak. `drop_duplicates()` runs
*before* the split, so a duplicate can never appear on both sides.

Result: **30,000 → 29,965 rows**.

## 3.4 The split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

| | Rows | Positive rate |
|---|---|---|
| Full dataset | 29,965 | 22.13% |
| Training (80%) | 23,972 | 22.13% |
| Test (20%) | 5,993 | 22.13% |

And inside training, for tuning:

```python
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)
```

→ fit 19,177 / validate 4,795.

**`stratify=y`** preserves the positive rate in every partition. **`random_state=42`**
makes the split reproducible. Both matter: without stratification a 20% sample has
non-trivial chance of landing at 20.5% or 21.7%, shifting every metric by noise.

## 3.5 Standardisation — the most important preprocessing step

### What it does

For each feature $j$, compute the mean $\mu_j$ and standard deviation $\sigma_j$ **on the
training data**, then transform every value:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

After this, every feature has mean 0 and standard deviation 1. A value of $z = 2$ means
"two standard deviations above average" **in any column**.

### Why it's mandatory

Look at the raw scales:

| Feature | Range | Contribution to squared distance |
|---|---|---|
| `LIMIT_BAL` | 10,000 – 1,000,000 | up to $10^{12}$ |
| `BILL_AMT3` | −157,264 – 1,664,089 | up to $10^{12}$ |
| `PAY_AMT2` | 0 – 1,684,259 | up to $10^{12}$ |
| `AGE` | 21 – 79 | about $10^{3}$ |
| `PAY_0` | −2 – 8 | about $1$ |

KNN computes Euclidean distance, SVM's RBF kernel is a function of Euclidean distance,
and logistic regression's gradient descent struggles when features differ by nine orders
of magnitude. **Without scaling, `PAY_0` — the single most predictive column — has a
smaller influence on distance than a rounding error in `BILL_AMT3`.**

Decision Tree and Naive Bayes don't care: a tree only compares against thresholds (and
any monotonic rescaling preserves the ordering), and NB models each feature
independently. We scale for everyone anyway so all five models see identical inputs —
which is what makes the comparison fair.

### The leakage trap

```python
scaler = StandardScaler().fit(X_tr)      # means/stds from FIT split only
X_tr_s    = scaler.transform(X_tr)
X_val_s   = scaler.transform(X_val)
X_train_s = scaler.transform(X_train)    # transform, never fit
X_test_s  = scaler.transform(X_test)     # transform, never fit
```

Calling `fit` on the **full** dataset before splitting would compute $\mu_j$ and
$\sigma_j$ from all 29,965 rows — including the test rows. The model would then
indirectly know the test set's distribution. This is **data leakage**: the test score
looks slightly better than the model deserves, and it is one of the most common silent
mistakes in student submissions.

Our final scaler is fit on 19,177 rows and applied to all four matrices with identical
parameters.

### Why the scaler must be saved

`model/scaler.joblib` is committed to the repo. When the Streamlit app receives an
uploaded CSV, it must divide by **the same** $\mu_j$ and $\sigma_j$ — not recompute them
from the upload. Recomputing would mean the uploaded rows are transformed on a different
scale than the training data, and every prediction would be meaningless. This is the
single most common deployment bug in this kind of project.

## 3.6 Artefacts the pipeline produces

| File | Contents | Why it's committed |
|---|---|---|
| `data/credit_default.csv` | 29,965 × 24 cleaned data | Reproduce training without the `.xls` |
| `test_data.csv` | 5,993 × 24 held-out rows | The assignment requires test data in the repo; also the app's default input |
| `model/scaler.joblib` | Fitted `StandardScaler` | Required at inference |
| `model/metadata.json` | Feature order, target name, positive class | **Column order must match training exactly** |

`metadata.json` matters more than it looks. `scaler.transform` applies column 0's mean
to column 0 of whatever array you hand it. If an uploaded CSV has the same columns in a
different order, the app will silently produce garbage predictions — no error, just wrong
answers. So the app does:

```python
X = load_scaler().transform(frame[FEATURES])   # FEATURES from metadata.json
```

…explicitly selecting and ordering columns rather than trusting the file's layout.

## 3.7 Class imbalance — what we did and didn't do

**What we did:** nothing to the class distribution. No oversampling, no undersampling, no
`class_weight`. The 22/78 split was left intact.

**Why:** resampling changes the meaning of predicted probabilities, and it makes the five
models harder to compare fairly. MCC and AUC are specifically designed to work under
imbalance, so we let the metrics handle it.

**What we could do** (all of these are legitimate improvements to discuss):

| Technique | Effect |
|---|---|
| `class_weight="balanced"` | Reweights the loss; usually raises recall substantially; few lines of code |
| Threshold tuning (§2.8) | Already demonstrated: 0.40 beats 0.50 on F1 and MCC |
| SMOTE oversampling | Synthesises minority rows; needs `imbalanced-learn` |
| Cost-sensitive learning | Encode that an FN costs ~5× an FP (the real ratio for a loan) |

Any of these would improve recall, at a precision cost. Leaving them out is a deliberate
scope choice, not an oversight — and saying so explicitly is worth more marks than
silently shipping a model with 0.37 recall.

Next: [04_code_walkthrough.md](04_code_walkthrough.md) for the line-by-line tour.
