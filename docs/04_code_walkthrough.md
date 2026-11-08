# 4. Code Walkthrough

Five scripts, executed in this order:

```
prepare_data.py   →  data/credit_default.csv
train_models.py   →  model/*.joblib, model/metadata.json, reports/*, test_data.csv
validate_models.py→  reports/validation.json      (checks our maths against sklearn)
analysis.py       →  reports/analysis.json        (intervals, threshold sweep)
app.py            →  Streamlit UI (reads the artefacts above, never retrains)
```

The five algorithms themselves live in `models/`, one file each, all subclassing a
common `Model` base class. That package is the heart of the project and gets its own
section (§4.3–4.4). scikit-learn appears in the pipeline for exactly three things —
`train_test_split`, `StandardScaler`, and the metric functions in `sklearn.metrics` — and
in `validate_models.py`, deliberately, as the reference implementation to check against.

---

## 4.1 `prepare_data.py` — 1 file in, 1 file out

```python
RAW = Path("data/raw/default of credit card clients.xls")
OUT = Path("data/credit_default.csv")
TARGET = "default"

df = pd.read_excel(RAW, header=1)
```

**`header=1`** — row 0 of the spreadsheet is a dummy header (`X1, X2, …`); the real column
names are on row 1. Getting this wrong silently produces `X1…X23` column names.

```python
df = df.drop(columns=["ID"])
df = df.rename(columns={"default payment next month": TARGET})
```

`ID` is 1…30000 — a pure row counter. Leaving it in would be dangerous: a tree could learn
splits like "ID < 15000" that appear predictive on training data purely by accident of
ordering. Renaming the target makes the rest of the code readable and satisfies the
assignment's `default` column contract.

```python
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})
```

`.replace({0: 4, 5: 4, 6: 4})` maps all three undocumented education codes into the
documented "others" bucket. Note this is `replace`, not `inplace` — the result is
reassigned.

```python
df = df.drop_duplicates().reset_index(drop=True)
```

30,000 → 29,965. `reset_index(drop=True)` renumbers 0…29,964 so later positional indexing
isn't lying.

```python
df.to_csv(OUT, index=False)
```

`index=False` prevents pandas' own row index becoming an unwanted first column — the
same class of bug as the `ID` column we just removed.

**Output:** `rows=29965 features=23`, `target rate=0.2213`.

---

## 4.2 `train_models.py` — the pipeline

### Imports and constants

```python
DATA          = Path("data/credit_default.csv")
MODEL_DIR     = Path("model")
REPORT_DIR    = Path("reports")
TEST_CSV      = Path("test_data.csv")
TARGET        = "default"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
SVM_MAX_SAMPLES = 4000
```

Paths are relative — **the script must be run from the project root**, because
`Path("data/...")` resolves against the current working directory. That's also why
Streamlit Cloud finds `model/` (there we used `Path(__file__).parent` instead, which is
absolute — see §4.7).

The import block is the clearest statement of the design:

```python
from models import (
    DecisionTreeModel, GaussianNBModel, KNNModel,
    LogisticRegressionModel, SVMModel,
)
```

There is no `from sklearn.linear_model import LogisticRegression`. The only sklearn
imports are `train_test_split`, `StandardScaler` and the metric functions — the parts of
the pipeline that are not the assignment's subject.

```python
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
```

A narrow suppression. On macOS, OpenBLAS emits spurious "divide by zero encountered in
matmul" warnings for denormal floating-point values. All outputs were verified finite
(`np.isfinite(...).all()` → `True`, probabilities in `[0,1]`), so this is noise, not
divergence. The `message=` pattern restricts the suppression to exactly that warning.

### `evaluate(model, X, y)`

```python
scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
preds = model.predict(X)
```

Two different quantities, deliberately:

- `predict_proba(X)[:, 1]` — probability of class 1. `[:, 1]` takes the second column of a
  `(n, 2)` array, i.e. P(default). Needed for AUC.
- `predict(X)` — hard 0/1 labels at threshold 0.5. Needed for precision, recall, F1, MCC.

All five hand-written models expose `predict_proba`, so the `hasattr` branch never fires
here; it stays as a guard so a future model that only produces a decision value still
scores correctly. `decision_function` returns a signed distance to the boundary — a valid
risk *ranking*, which is all AUC needs (see `01_theory.md` §1.7).

```python
return {
    "accuracy":  accuracy_score(y, preds),
    "auc":       roc_auc_score(y, scores),
    "precision": precision_score(y, preds),
    "recall":    recall_score(y, preds),
    "f1":        f1_score(y, preds),
    "mcc":       matthews_corrcoef(y, preds),
}
```

Note the positive class defaults to `1` for all of these — which is what we want, since
"default" is the event of interest. If the target had been coded 1 = "good customer", every
precision/recall number would be about the opposite class.

### `tune(name, build, grid, ...)`

```python
best_params, best_score = None, float("-inf")
for params in grid:
    model = build(**params).fit(X_tr, y_tr)
    score = f1_score(y_val, model.predict(X_val))
    if score > best_score:
        best_params, best_score = params, score
return best_params
```

A hand-rolled grid search, ~10 lines instead of `GridSearchCV`. It:

- builds a **fresh** model from each parameter dict,
- fits it on the **fit split** (19,177 rows),
- scores it on the **validation split** (4,795 rows) using F1,
- returns the winning *parameters*, not the winning *model*.

That last detail matters more than it looks. The function could have returned the fitted
estimator, and an earlier version of this project did — which is exactly how the bug in
§4.9 happened. Returning parameters makes it structurally impossible to accidentally keep
a model that was trained on only 80% of the training data.

**Why F1 and not accuracy?** Accuracy would pick the tree that predicts "no default" most
often, which is the wrong objective on a 22%-positive problem. F1 balances the two error
types.

`-inf` as the initial best means the first candidate always wins the initial comparison,
however low its score.

### `build_models(tree_params, knn_params)`

```python
return {
    "logistic_regression": LogisticRegressionModel(C=1.0),
    "decision_tree":       DecisionTreeModel(**tree_params),
    "knn":                 KNNModel(**knn_params),
    "naive_bayes":         GaussianNBModel(),
    "svm":                 SVMModel(C=1.0, max_samples=SVM_MAX_SAMPLES,
                                   random_state=RANDOM_STATE),
}
```

Five unfitted objects, one dict, one shared interface. Everything downstream — the fit
loop, `evaluate`, `model.save`, the Streamlit app — works without knowing which algorithm
it is holding.

### `main()` step by step

**Step 1 — load and split**

```python
frame = pd.read_csv(DATA)
features_frame = frame.drop(columns=[TARGET])
feature_names = list(features_frame.columns)
y = frame[TARGET].to_numpy(int)

train_index, test_index = train_test_split(
    np.arange(len(frame)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
X_train = features_frame.iloc[train_index].to_numpy(float)
X_test  = features_frame.iloc[test_index].to_numpy(float)
y_train, y_test = y[train_index], y[test_index]

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)
```

Two choices are deliberate here.

**Row indices are split, not arrays.** `train_test_split` on `np.arange(29965)` returns the
same permutation it would have returned for the feature matrix, but keeping the indices
lets the held-out CSV be rebuilt from the *original* frame in §Step 5 — so integer columns
stay integers instead of every value acquiring a `.0`.

**The array conversion happens after the split.** The models do arithmetic on `X`
directly, so they get a float array; pandas stays in charge of dtypes and column names
right up to that boundary. `y` is an integer array because the models use it for `y == c`
comparisons and for indexing, never for pandas lookups.

The second split is 20% of the *training* set, not of the whole dataset. `stratify=y` on
both splits keeps the 22% positive rate identical in all three partitions; without it a
small validation split could drift to 20% or 25% and make the tuning comparison noisy.

**Step 2 — scale**

```python
scaler = StandardScaler().fit(pd.DataFrame(X_train, columns=feature_names))
X_tr_s, X_val_s     = scaler.transform(X_tr), scaler.transform(X_val)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
```

`fit` **once**, on the full training partition, then `transform` everywhere else. Fitting
on `X_test` would leak the test distribution into training; fitting only on `X_tr` would
throw away the val split's contribution to the scale estimate. Fitting on all 23,972
training rows and then transforming the tuning split with those same parameters is the
correct order — the validation rows are part of training data.

The scaler is fitted through a **DataFrame**, not the raw array, so it records the 23
column names. The app later calls `load_scaler().transform(frame[FEATURES])` with a
DataFrame; if the scaler had been fitted on an array, pandas and sklearn would both warn
about mismatched feature names.

**Step 3 — define the tuning grids**

```python
tree_params = tune("decision_tree", DecisionTreeModel,
                   [{"max_depth": d, "min_samples_leaf": 50}
                    for d in (4, 6, 8, 10, None)],
                   X_tr_s, y_tr, X_val_s, y_val)
knn_params = tune("knn", KNNModel,
                  [{"n_neighbors": k, "weights": "distance"}
                   for k in (15, 25, 45, 75)],
                  X_tr_s, y_tr, X_val_s, y_val)
```

`min_samples_leaf=50` is held constant across all tree candidates so the only thing
varying is depth — one variable at a time. `max_depth=None` means "grow until the leaves
are pure or hit the leaf-size floor"; `DecisionTreeModel._grow` handles it with an
explicit `is not None` check rather than assuming an integer.

Nine validation fits total (5 trees + 4 KNN). Cheap: trees are milliseconds, KNN `fit` is
just storing the matrix.

**Step 4 — fit every model on the full training set**

```python
models = build_models(tree_params, knn_params)
...
for name, model in models.items():
    model.fit(X_train_s, y_train)
    metrics = evaluate(model, X_test_s, y_test)
    ...
    model.save(MODEL_DIR / f"{name}.joblib")
```

Every model is built fresh from its winning parameters and fitted on all **23,972** rows.
There is no `clone()` call because there is nothing to clone: `tune()` never handed back a
fitted model. Uniform treatment also means the three untuned models aren't special-cased.

One model quietly does less work than the others: `SVMModel.fit` subsamples internally.
Its precomputed kernel matrix is *n*×*n*, which at 23,972 rows would be ~4.6 GB, so it
draws a stratified 4,000-row subset (`_subsample`) and solves on that. The compromise is
visible in the results — SVM AUC 0.7105, the lowest of the five — and it is stated rather
than hidden.

`y_test` is used here and **only** here. This is the one moment the sealed partition is
opened.

**Step 5 — persist artefacts**

```python
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
(MODEL_DIR / "metadata.json").write_text(
    json.dumps({"features": feature_names, "target": TARGET, "positive_class": 1}, indent=2))

test_out = features_frame.iloc[test_index].copy()
test_out[TARGET] = y_test
test_out.to_csv(TEST_CSV, index=False)
```

`feature_names` records the exact 23 column names **in training order**. This is the
contract the app depends on.

`test_out` is taken from the original frame by row index rather than from the scaled
matrix, so the committed CSV contains the *unscaled* features — `LIMIT_BAL` is
`240000`, not `2.41` — and keeps its integer dtypes. Anyone can open `test_data.csv`,
add rows, and score them through the app, because the app applies the saved scaler
itself.

`reports/training_config.json` records the chosen hyperparameters and split sizes, so a
result can be traced back to the configuration that produced it.

### Expected output

```
Tuning the two hyperparameters that matter most:
  decision_tree  {'max_depth': 4, 'min_samples_leaf': 50}  val_f1=0.4605
  decision_tree  {'max_depth': 6, 'min_samples_leaf': 50}  val_f1=0.4609
  decision_tree  {'max_depth': 8, 'min_samples_leaf': 50}  val_f1=0.4432
  decision_tree  {'max_depth': 10, 'min_samples_leaf': 50}  val_f1=0.4323
  decision_tree  {'max_depth': None, 'min_samples_leaf': 50}  val_f1=0.4323
  knn            {'n_neighbors': 15, 'weights': 'distance'}  val_f1=0.4365
  knn            {'n_neighbors': 25, 'weights': 'distance'}  val_f1=0.4218
  ...
logistic_regression  accuracy=0.8133  auc=0.7160  precision=0.7285  recall=0.2489  f1=0.3710  mcc=0.3494
decision_tree        accuracy=0.8195  auc=0.7472  precision=0.6658  recall=0.3695  f1=0.4753  mcc=0.4007
...
```

Depth 4 and depth 6 are separated by 0.0004 in validation F1 — a tie in any practical
sense. The tuner takes the larger value and picks depth 6; the honest way to read that row
is "shallow trees work, and anything deeper than 6 is strictly worse."

---

## 4.3 `models/base.py` — why an abstract base class

```python
class Model(ABC):
    name = "model"

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict_proba(self, X): ...

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
```

Only `fit` and `predict_proba` are abstract: those are the two things that are genuinely
different in every algorithm. Everything else is shared:

| Member | What it does | Why it is on the base class |
|---|---|---|
| `predict(X, threshold)` | threshold `predict_proba` at 0.5 | turning a probability into a label is a policy decision, not an algorithm — and it is the one place a threshold change belongs (`analysis.py` sweeps it) |
| `save(path)` / `load(path)` | `joblib.dump` / `joblib.load` | same serialisation for all five, and pickle can restore any subclass because the class path is stored in the file |
| `_validate_X(X)` | coerce to float, check 2-D, check feature count | catches the single most common serving bug — a column silently dropped or reordered — at the boundary instead of deep inside a matrix multiply |
| `_sigmoid(z)` | `1 / (1 + e^-z)`, clipped to ±500 | used by logistic regression and by the SVM's pseudo-probabilities; the clip stops `exp` overflow |

The ABC also enforces the contract at construction time: a subclass that forgets
`predict_proba` cannot be instantiated, so the failure surfaces when you write the class,
not three scripts later.

The interface is deliberately the *same shape* as scikit-learn's. That is what lets
`validate_models.py` drop either implementation into the same `scores()` function and
compare like for like.

**What every `fit` must set:** `n_features_in_` (so `_validate_X` can check later input)
and `classes_` (so `predict` knows which label is which). Those two attributes are the
informal half of the contract, documented in the docstring.

---

## 4.4 The five implementations

Theory lives in `01_theory.md`; this section is about where to look in the code and why it
is written that way.

### Logistic Regression — `models/logistic.py`

The model has 24 parameters: 23 weights plus an intercept. `fit` runs Newton's method:

```python
p = self._sigmoid(X @ beta[1:] + beta[0])
residual = p - y
grad[0] = self.C * residual.sum()
grad[1:] = beta[1:] + self.C * (X.T @ residual)
...
hessian[1:, 1:] = np.eye(d) + self.C * (X.T @ (X * w[:, None]))
step = np.linalg.solve(hessian + 1e-10 * np.eye(d + 1), grad)
beta -= step
```

Three details worth reading twice:

- `grad[0]` is the intercept's gradient and has **no** `beta[0]` term — the intercept is
  not penalised, matching sklearn's convention.
- `hessian[1:, 1:]` adds `np.eye(d)`, the L2 penalty's second derivative. Same for the
  weights in `grad[1:]`.
- `1e-10 * np.eye(...)` is a numerical crutch, not a modelling choice: if a feature is
  constant on the training split its column of the Hessian is zero and the matrix is
  singular.

Convergence is checked on the *step*, not the loss: when `max|Δβ| < tol`, the parameters
have stopped moving. Here it takes 7 iterations.

`predict` overrides the base version and thresholds the decision value at 0 — equivalent
to thresholding the probability at 0.5, but avoids computing a sigmoid only to undo it.

### Decision Tree — `models/tree.py`

Each node is a dict: `{"prob", "n", "impurity"}`, plus `{"feature", "threshold", "left",
"right"}` if it splits. A leaf is simply a node without `"feature"`, which is what
`_apply` tests:

```python
def _apply(self, node, X, rows, out):
    if "feature" not in node:
        out[rows] = node["prob"]
        return
    go_left = X[rows, node["feature"]] <= node["threshold"]
    self._apply(node["left"], X, rows[go_left], out)
    self._apply(node["right"], X, rows[~go_left], out)
```

This is vectorised-by-subsetting rather than one row at a time: each recursion passes
*indices*, so a node with 20,000 rows still does one vectorised comparison. `rows[go_left]`
and `rows[~go_left]` partition the index set exactly, so every row reaches exactly one leaf.

The split search is where the real work is. For each feature:

```python
order = np.argsort(column, kind="mergesort")
sorted_col = column[order]
cumulative_pos = np.cumsum(pos[order])
idx = np.searchsorted(sorted_col, candidates, side="right")
left_n, right_n = idx.astype(float), n - idx.astype(float)
left_pos = cumulative_pos[idx - 1]
```

Sorting once per feature turns "what if we cut here?" into a prefix-sum lookup. For every
candidate threshold, `idx` is how many rows fall left, and `cumulative_pos[idx - 1]` is how
many of those are positive. That gives both sides' impurity in a few vectorised operations
instead of re-partitioning the data for each candidate.

`side="right"` matters: it counts rows `<= threshold` as left, matching the `<=` used in
`_apply`. Using the wrong side here would train on one split rule and predict with another.

Candidates themselves are `max_thresholds=32` quantiles when a feature has many distinct
values, and exact midpoints when it has few. Quantiles are the one deliberate approximation
versus scikit-learn, and `validate_models.py` is the measurement of what it costs.

`_importances()` walks the tree and accumulates `n · (impurity_node − weighted child
impurity)` per feature, then normalises. That is the same definition sklearn uses, which is
why the README can quote `PAY_0` at 66.7%.

### K-Nearest Neighbours — `models/knn.py`

`fit` stores the data and precomputes `_X_sq = Σ x²` per training row. Prediction expands
the squared distance:

```python
d2 = x_sq[start:stop, None] - 2.0 * (block @ self._X.T) + self._X_sq[None, :]
```

`‖x − x'‖² = ‖x‖² − 2x·x' + ‖x'‖²`. The middle term is the only one that couples rows and
columns, so it becomes a single matrix multiply; the two norm terms are already known. This
is the difference between a Python loop over 24,000 pairs and one BLAS call.

`np.maximum(d2, 0.0, out=d2)` deals with the fact that the expansion is algebraically exact
but numerically not: cancellation can leave `-1e-16` where the true value is 0.

```python
neighbours = np.argpartition(d2, k - 1, axis=1)[:, :k]
```

`argpartition` puts the *k* smallest in the first *k* positions in O(n) per row — no full
sort. Since only the set of neighbours matters (not their order), sorting 24,000 values to
read the first 15 would be wasted work.

Distance weights use `np.where(dist <= 1e-12, 1e12, 1.0 / np.maximum(dist, 1e-12))`: an
exact duplicate must dominate the vote, and `1/0` would be `inf` and poison the sum.

Rows are processed in chunks of 512 because a full `(5993 × 23972)` float64 distance matrix
is ~1.15 GB.

### Gaussian Naive Bayes — `models/naive_bayes.py`

`fit` is four lines of statistics: per-class means (`theta_`), per-class variances
(`var_`), a smoothing term, and the priors. The smoothing:

```python
epsilon = self.var_smoothing * X.var(axis=0).max()
self.var_ = self.var_ + epsilon
```

Without it, a feature that is constant within one class has variance 0, and the
`(x − μ)² / σ²` term divides by zero. The added value is scaled by the largest observed
variance so one constant feature cannot distort features on a different scale.

Everything else happens in `_joint_log_likelihood`, which returns
`log P(y) + Σ log P(x_i | y)` per class. Multiplying 23 Gaussian densities underflows to 0
in float64; summing 23 logs does not. `predict_proba` then subtracts the row maximum before
`exp` — the standard log-sum-exp trick — so the largest log-likelihood maps to 1.0 and the
rest are relative to it.

`predict` overrides the base class to take `argmax` of the log-likelihood rather than
thresholding a probability. Identical answer, no exponentiating.

### Support Vector Machine — `models/svm.py`

The class holds the whole SMO loop in `fit`. The parts that repay reading:

```python
eta = 2.0 * kernel[i, j] - kernel[i, i] - kernel[j, j]
if eta >= -1e-12:
    continue
```

`eta` is the second derivative of the objective along the `(i, j)` direction. For a valid
kernel it is negative; the guard skips the pair when it is not, because dividing by
`eta ≈ 0` would send the multiplier to infinity. This is also why custom kernels must be
positive semi-definite — SMO assumes it.

```python
alphas[j] = alpha_j_old - t[j] * (error_i - error_j) / eta
alphas[j] = min(high, max(low, alphas[j]))
```

One closed-form step, then a clip to `[low, high]` — the box constraints combined with the
equality constraint `Σ α_i t_i = 0`. Clipping is not a heuristic; it is what keeps the
solution feasible.

`f` is a cache of the current decision value for every training point. Recomputing
`Σ α_j t_j K(i, j) + b` for each check would make a sweep O(n²) *inside* the O(n) loop;
instead the two changed terms are applied to all rows at once:

```python
f += delta_i * kernel[i] + delta_j * kernel[j] + (new_b - b)
```

That `(new_b - b)` term is easy to forget, and forgetting it is a real bug we hit: `f` is
the *full* decision value, so when the bias moves, every cached value moves with it. Without
it the errors drift and the intercept diverges — our first version converged to `b ≈ 114`
and predicted one class everywhere. The fix is one term; finding it required comparing
against sklearn.

```python
support = alphas > 1e-8
self.dual_coef_ = (alphas * t)[support]
```

For prediction only the support vectors are kept: 1,877 of the 4,000 subsampled rows.
`decision_function` evaluates the kernel against those, chunked to bound memory, and
`predict_proba` wraps it in a sigmoid. That sigmoid is monotone, so the *ranking* — and
therefore AUC — is exact, but the probabilities are not calibrated. Anything that treats
them as real probabilities (pricing, expected loss) should not use this model.

---

## 4.5 `validate_models.py` — the correctness check

This script exists to answer one question: **is the hand-written maths actually the
algorithm it claims to be?** It fits each of our models and scikit-learn's reference
version on the *identical* split and prints the metric difference.

```python
for name, model in (("sklearn", reference), ("from-scratch", ours)):
    started = time.perf_counter()
    model.fit(X_fit, y_fit)
    elapsed = time.perf_counter() - started
    result = scores(model, X_eval, y_true)
```

`scores()` relies only on the shared interface — `predict_proba` (or `decision_function`),
`predict` — which is why the same function can score both implementations. That shared
shape is the payoff from §4.3.

Two comparison details worth noting:

- **The means and variances for scaling are fitted once**, on the identical training rows,
  and applied to both. If each fitting path computed its own scaler the comparison would
  measure two different transforms, not two implementations.
- **The SVM comparison subsamples first.** `svm_ours._subsample(...)` is called once to
  produce a stratified 4,000-row subset, and *that same subset* is passed to both sklearn's
  `SVC` and our model. Our `fit` then calls `_subsample` again, but 4,000 ≤ 4,000 so it is
  a no-op. This keeps the comparison honest: same rows, same kernel, same `gamma`.

Results (printed and written to `reports/validation.json`):

| Model | Largest metric gap | Why any gap exists |
|---|---|---|
| Logistic Regression | AUC −0.0000 | Newton and lbfgs reach the same optimum |
| Decision Tree | AUC −0.0035 | quantile thresholds vs exhaustive midpoints |
| KNN | AUC +0.0002 | tie-breaking among equal-distance neighbours |
| Naive Bayes | AUC 0.0000 | closed-form estimates, no solver involved |
| SVM | AUC −0.0001 | different SMO working-set selection |

The decision tree's gap is the only one worth explaining, and it is an *approximation* gap,
not a bug: with 32 quantile candidates per feature the search cannot land on a split
between them. Three thousandths of an AUC buys a training time that is minutes instead of
hours.

`validate_models.py` fits the tree at `max_depth=4`, matching the reference configuration,
not the tuned depth 6. It is testing the implementation, not re-running the experiment.

---

## 4.6 `analysis.py` — the numbers behind the caveats

The README makes two kinds of claim that a single test-set score cannot support: that a
model is "better" than another, and that 0.5 is the right threshold. This script produces
the evidence for both.

**Bootstrap.** Resample the 5,993 test rows with replacement, recompute AUC and MCC, repeat
300 times, take the 2.5th and 97.5th percentiles:

```python
resamples = [rng.integers(0, n, n) for _ in range(N_BOOTSTRAP)]
```

One list of index vectors is built and reused for every model, so the resamples are
*paired* across models — a difference between two models is measured on the same synthetic
datasets, which is the right comparison.

```python
if y[index].min() == y[index].max():
    continue
```

A resample that happens to contain only one class has no defined AUC. Skipping it is
correct; clamping it would silently bias the interval.

The outcome is the README's central caveat: most intervals overlap, so most pairwise
"wins" are sampling noise. Only two claims hold — the tree's MCC over Naive Bayes, and its
AUC over the SVM.

**Threshold sweep.** `predict` thresholds at 0.5 because `Model.predict` defaults to it.
That default is a convention, not an optimum. The sweep recomputes precision, recall, F1
and MCC at six cut-offs from the stored probabilities and finds F1 peaking near 0.25 while
MCC peaks at 0.50 — different metrics encode different costs, so "the best threshold" is a
business question the data cannot answer alone.

**Importances.** Not recomputed here; read from `tree.feature_importances_`, which
`DecisionTreeModel._importances()` filled during `fit`. The script only sorts and prints.

Everything is written to `reports/analysis.json`.

---

## 4.7 `app.py` — inference only

### Why no training code appears here

The app imports `joblib`, `pandas`, `matplotlib` and metrics — but **never**
`train_test_split`, `StandardScaler`-fit, or any of the `models/` classes directly. That is
a design guarantee required by the assignment: *"The app must load the saved model,
calculate the evaluation metrics, and display the results without retraining the model."*
If you see a `.fit(` anywhere in this file, something has gone wrong.

There is one subtlety: `joblib.load` restores the class by importing
`models.<module>.<Class>` from the pickle. The app never writes that import, but the
package must ship alongside `app.py` or loading fails — a deployment detail worth knowing.

### Path handling

```python
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
BUNDLED_TEST_DATA = ROOT / "test_data.csv"
```

`Path(__file__).parent` is absolute — it resolves relative to the **script's** location, not
the shell's working directory. Streamlit Cloud does not launch from a predictable cwd, so
`train_models.py`'s relative `Path("model")` style would break here. Different context,
different rule.

### Caching

```python
@st.cache_resource
def load_model(key):
    return joblib.load(MODEL_DIR / f"{key}.joblib")
```

Streamlit reruns the **entire script top-to-bottom on every widget interaction**. Without
caching, switching the dropdown would re-read the 4.6 MB KNN artefact from disk each time.
`st.cache_resource` keeps one copy in memory across reruns, keyed by `key`. This is why
switching models in the UI feels instant.

### Upload handling

```python
if upload is not None:
    frame = pd.read_csv(upload)
    source = upload.name
else:
    frame = pd.read_csv(BUNDLED_TEST_DATA)
    source = "bundled test_data.csv"
```

`upload` is an in-memory file object — the user's CSV is never written to disk. The
fallback to the bundled test set means the app is demonstrable immediately on first load,
before anyone uploads anything.

### Validation and the column-order guarantee

```python
missing = [column for column in FEATURES if column not in frame.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required column(s): ...")
    st.stop()

model = load_model(MODEL_CHOICES[model_label])
X = load_scaler().transform(frame[FEATURES])
```

Three things happen in that last line, all essential:

1. **`frame[FEATURES]`** selects columns **by name, in training order** — so a CSV with the
   right columns in a scrambled order still scores correctly.
2. **`load_scaler().transform(...)`** applies the *persisted* training means and standard
   deviations. It does not fit a new scaler on the uploaded data. If it did, the same client
   would be scored differently depending on who else happened to be in the file.
3. **`_validate_X` inside the model** checks the width a second time, so a mismatch fails
   with a clear message instead of a broadcasting error.

`st.stop()` halts the script — without it, the subsequent `frame[FEATURES]` would raise a
`KeyError` and show a stack trace to the user.

### Scoring and rendering

```python
results, preds = score(model, X, y)
columns = st.columns(6)
for column, (key, label) in zip(columns, METRIC_LABELS.items()):
    column.metric(label, f"{results[key]:.4f}")
```

`score()` mirrors `evaluate()` from `train_models.py` — same six metrics, same
`predict_proba` vs `decision_function` branch — except it also returns `preds` so the
confusion matrix can be drawn without predicting twice. It adds `zero_division=0` because
an uploaded CSV might contain no positives at all, which would otherwise produce a warning
and a `nan`.

```python
st.caption(
    "Accuracy alone is misleading: predicting 'no default' for everyone "
    f"already scores {(y == 0).mean():.4f} on this data.")
```

The majority-class baseline, computed live from whatever was uploaded. This is the app
deliberately refusing to let a bare accuracy number look impressive.

```python
def confusion_figure(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ...
    shade = "white" if value > matrix.max() / 2 else "#1f2933"
```

`labels=[0, 1]` forces a 2×2 even if the upload happens to contain only one class.
The `shade` line flips the text colour on dark-blue cells so the counts stay readable —
the cell showing `TN = 4,421` would otherwise be dark text on a dark square.

`matplotlib.use("Agg")` selects the non-interactive backend. On a headless server there is
no display to draw into; without `Agg`, plotting can fail at import time. The `# noqa: E402`
comment suppresses the linter's "import not at top of file" complaint, because
`matplotlib.use()` must run **before** `pyplot` is imported.

---

## 4.8 Data flow, end to end

```
credit card .xls
      │  pd.read_excel(header=1)
      ▼
30,000 × 25 DataFrame
      │  drop ID, fold codes, drop_duplicates
      ▼
29,965 × 24  →  data/credit_default.csv
      │  train_test_split(0.2, stratify, seed 42)
      ├──────────────────────────────┐
      ▼                              ▼
23,972 train                    5,993 test (sealed)
      │  split 80/20 for tuning        │
      ├── 19,177 fit                   │
      ├──  4,795 validation            │
      │                                │
      │  tune → params → build fresh → refit on 23,972
      ▼                                ▼
5 fitted models ──────────────► score on test
      │                                │
      │  model.save()                  ▼
      ▼                        reports/metrics.json
model/*.joblib                 reports/details.json
model/scaler.joblib            reports/training_config.json
model/metadata.json
      │
      ▼
app.py  ── loads artefacts, transforms uploads with the saved scaler, predicts
      │
      └─ validate_models.py  (independent check against sklearn)
         analysis.py        (bootstrap intervals, threshold sweep)
```

---

## 4.9 A real bug that was in this project

Worth studying, because it is a subtle one, it changed the reported numbers, and the
current design is shaped by it.

**The first version** had `tune()` return the *fitted* estimator, and the main loop skipped
refitting the models it had already fitted:

```python
models["decision_tree"] = tune(...)   # fitted on X_tr — 19,177 rows
...
for name, model in models.items():
    if name not in ("decision_tree", "knn"):   # ← wrong
        model.fit(X_train_s, y_train)
```

So the Decision Tree and KNN were scored having been denied 20% of their training data,
purely because they were the two models whose hyperparameters we had tuned.

**How it was caught:** inspecting artefacts rather than trusting the printed metrics —

```python
knn = joblib.load("model/knn.joblib")
knn._fit_X.shape        # → (19177, 23)   ← should have been (23972, 23)
dt.get_n_leaves()       # → 13            ← too few for 23,972 rows at depth 4
```

**The fix at the time** was `clone(model)` before refitting — keep the hyperparameters,
discard the small fit:

```python
if name in ("decision_tree", "knn"):
    model = clone(model)
model.fit(X_train_s, y_train)
```

**The fix that stuck** was structural: `tune()` now returns a parameter dict instead of a
model, and `build_models()` constructs every estimator fresh. There is no fitted object
that could be accidentally reused, so the bug cannot recur regardless of how the loop is
written.

**The lesson:** a model's *hyperparameters* are chosen on validation data, but the final
model should always be retrained on everything available. Keeping the model trained on the
smaller split silently throws data away. `GridSearchCV(refit=True)` is the library's default
for exactly this reason.

A second, smaller bug in the same family: the first SVM implementation updated the cached
decision values `f` for changes in the multipliers but not for changes in the bias, so the
intercept drifted to 114 and the model predicted one class everywhere. It was caught only
by fitting scikit-learn's SVM on the same data and seeing the disagreement — which is the
argument for having `validate_models.py` at all.

---

## 4.10 Running it

```bash
python prepare_data.py      # ~2 s   → data/credit_default.csv
python train_models.py      # ~15 s  → model/, reports/, test_data.csv
python validate_models.py   # ~15 s  → reports/validation.json
python analysis.py          # seconds → reports/analysis.json
streamlit run app.py        # opens http://localhost:8501
```

`train_models.py` is dominated by the SVM: the SMO loop runs 3,471 sweeps over the
4,000-row subsample and keeps 1,877 support vectors. `validate_models.py` costs about the
same because it trains both implementations of every algorithm.

Next: [05_why_these_results.md](05_why_these_results.md) ties every number back to a
mechanism.
