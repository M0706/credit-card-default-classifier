# 4. Code Walkthrough

Three scripts, executed in this order:

```
prepare_data.py  →  data/credit_default.csv
train_models.py  →  model/*.joblib, model/metadata.json, reports/*, test_data.csv
app.py           →  Streamlit UI (reads the artefacts above, never retrains)
```

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

## 4.2 `train_models.py` — the core

### Imports and constants (lines 1–37)

```python
DATA        = Path("data/credit_default.csv")
MODEL_DIR   = Path("model")
REPORT_DIR  = Path("reports")
TEST_CSV    = Path("test_data.csv")
TARGET      = "default"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
```

Paths are relative — **the script must be run from the project root**, because
`Path("data/...")` resolves against the current working directory. That's also why
Streamlit Cloud finds `model/` (there we used `Path(__file__).parent` instead, which is
absolute — see §4.4).

```python
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
```

A narrow suppression. On macOS, OpenBLAS emits spurious "divide by zero encountered in
matmul" warnings for denormal floating-point values. All outputs were verified finite
(`np.isfinite(...).all()` → `True`, probabilities in `[0,1]`), so this is noise, not
divergence. The `message=` pattern restricts the suppression to exactly that warning.

### `evaluate(model, X, y)` (lines 40–50)

```python
scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
preds = model.predict(X)
```

Two different quantities, deliberately:

- `predict_proba(X)[:, 1]` — probability of class 1. `[:, 1]` takes the second column of a
  `(n, 2)` array, i.e. P(default). Needed for AUC.
- `predict(X)` — hard 0/1 labels at threshold 0.5. Needed for precision, recall, F1, MCC.

The `hasattr` branch exists solely for `SVC`, which has no `predict_proba` unless built
with `probability=True`. `decision_function` returns the signed distance to the separating
boundary — a valid risk *ranking*, which is all AUC needs (see `01_theory.md` §1.7).

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

### `tune(name, candidates, ...)` (lines 53–61)

```python
best_score, best_model = -np.inf, None
for params, model in candidates:
    model.fit(X_tr, y_tr)
    score = f1_score(y_val, model.predict(X_val))
    if score > best_score:
        best_score, best_model = score, model
return best_model
```

A hand-rolled grid search, ~10 lines instead of `GridSearchCV`. It:

- fits each candidate on the **fit split** (19,177 rows),
- scores it on the **validation split** (4,795 rows) using F1,
- returns the best *fitted* estimator.

**Why F1 and not accuracy?** Accuracy would pick the tree that predicts "no default" most
often, which is the wrong objective on a 22%-positive problem. F1 balances the two error
types.

`-np.inf` as the initial best means the first candidate always wins the initial comparison,
however low its score. (`float('-inf')` would work identically.)

### `build_models()` (lines 64–71)

```python
"logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
"decision_tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
"knn":                 KNeighborsClassifier(),
"naive_bayes":         GaussianNB(),
"svm":                 SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
```

`max_iter=2000` for LR: the default 100 raises a `ConvergenceWarning` on this data (it
converged in 32, so the ceiling is just headroom). The Decision Tree and KNN entries here
are placeholders — both get overwritten by `tune()` on the next lines. `gamma="scale"`
defaults to `1 / (n_features × Var(X))`, which adapts to our 23-feature scale.

### `main()` step by step

**Step 1 — load and split (lines 75–84)**

```python
df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE)
```

`X` is a DataFrame (23 columns incl. names); `y` is a Series. The second split is 20% of
the *training* set, not of the whole dataset.

**Step 2 — scale (lines 86–88)**

```python
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_val_s   = scaler.transform(X_tr), scaler.transform(X_val)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
```

`fit` **once**, on `X_tr` only. Everything else is `transform`. That single line is the
difference between a valid score and a leaked one (see `03_data_pipeline.md` §3.5).

**Step 3 — define the tuning grids (lines 91–98)**

```python
tree_candidates = [
    ({"max_depth": d},
     DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, random_state=RANDOM_STATE))
    for d in (4, 6, 8, 10, None)
]
knn_candidates = [
    ({"n_neighbors": k}, KNeighborsClassifier(n_neighbors=k, weights="distance"))
    for k in (15, 25, 45, 75)
]
```

Each entry is `(params_for_logging, unfitted_estimator)`. `min_samples_leaf=50` is held
constant across all tree candidates so the only thing varying is depth — one variable at a
time.

**Step 4 — tune (lines 100–102)**

```python
models = build_models()
models["decision_tree"] = tune("decision_tree", tree_candidates, X_tr_s, y_tr, X_val_s, y_val)
models["knn"]          = tune("knn", knn_candidates, X_tr_s, y_tr, X_val_s, y_val)
```

Nine validation fits total (5 trees + 4 KNN). Cheap: trees are milliseconds, KNN `fit` is
just storing the matrix.

**Step 5 — refit and score (lines 107–122)**

```python
for name, model in models.items():
    if name in ("decision_tree", "knn"):
        model = clone(model)
    model.fit(X_train_s, y_train)
    metrics = evaluate(model, X_test_s, y_test)
    ...
    preds = model.predict(X_test_s)
    reports[name] = {
        "classification_report": classification_report(y_test, preds, digits=4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
```

`clone(model)` returns a **fresh, unfitted copy with the same hyperparameters**. For the
tuned models this is what discards the 19,177-row fit so the model can be retrained on all
23,972 rows. For the other three, `clone` isn't needed — they were never fitted.

`y_test` is used here and **only** here. This is the one moment the sealed partition is
opened.

**Step 6 — persist artefacts (lines 124–134)**

```python
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
(MODEL_DIR / "metadata.json").write_text(
    json.dumps({"features": list(X.columns), "target": TARGET, "positive_class": 1}, indent=2))

test_out = X_test.copy()
test_out[TARGET] = y_test.values
test_out.to_csv(TEST_CSV, index=False)
```

`list(X.columns)` records the exact 23 column names **in training order**. This is the
contract the app depends on. `X_test.copy()` avoids modifying the DataFrame we still
might want, and `.values` aligns the target by position.

### Expected output

```
Tuning the two scale-sensitive hyperparameters:
  decision_tree      {'max_depth': 4}  val_f1=0.4628
  decision_tree      {'max_depth': 6}  val_f1=0.4517
  decision_tree      {'max_depth': 8}  val_f1=0.4449
  decision_tree      {'max_depth': 10}  val_f1=0.4377
  decision_tree      {'max_depth': None}  val_f1=0.4290
  knn                {'n_neighbors': 15}  val_f1=0.4375
  ...
logistic_regression  accuracy=0.8133  auc=0.7160  precision=0.7275  recall=0.2496  f1=0.3717  mcc=0.3496
decision_tree        accuracy=0.8163  auc=0.7341  precision=0.6490  recall=0.3695  f1=0.4709  mcc=0.3912
...
```

---

## 4.3 A real bug that was in this project

Worth studying, because it is a subtle one and it changed the reported numbers.

**The first version** looked like this:

```python
for name, model in models.items():
    if name not in ("decision_tree", "knn"):   # ← wrong
        model.fit(X_train_s, y_train)
```

The intent was "don't refit models that `tune()` already fitted". But `tune()` fits on
`X_tr` — only **19,177 of 23,972** training rows. So the Decision Tree and KNN were
scored while having been denied 20% of their training data, purely because they were the
two models whose hyperparameters we had tuned.

**How it was caught:** inspecting artefacts rather than trusting the printed metrics —

```python
knn = joblib.load("model/knn.joblib")
knn._fit_X.shape        # → (19177, 23)   ← should have been (23972, 23)
dt.get_n_leaves()       # → 13            ← too few for 23,972 rows at depth 4
```

**The fix:**

```python
if name in ("decision_tree", "knn"):
    model = clone(model)        # keep hyperparameters, discard the small fit
model.fit(X_train_s, y_train)   # everyone trains on the full set
```

**The effect:** Decision Tree MCC 0.3858 → **0.3912**, F1 0.4623 → 0.4709. KNN MCC
0.3611 → 0.3545 (it got slightly *worse* — a larger reference set changes which 15
neighbours vote near the boundary; both outcomes are legitimate, which is why you measure
rather than assume).

**The general lesson:** a model's *hyperparameters* are chosen on validation data, but the
final model should always be retrained on everything available. If you keep the model
trained on the smaller split, you silently throw data away. `GridSearchCV(refit=True)` is
the library's default for exactly this reason.

---

## 4.4 `app.py` — inference only

### Why no training code appears here

The app imports `joblib`, `pandas`, `matplotlib` and metrics — but **never** `LogisticRegression`,
`DecisionTreeClassifier`, `train_test_split` or `StandardScaler`-fit. That is a design
guarantee required by the assignment: *"The app must load the saved model, calculate the
evaluation metrics, and display the results without retraining the model."* If you see a
`.fit(` anywhere in this file, something has gone wrong.

### Path handling (lines 33–35)

```python
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
BUNDLED_TEST_DATA = ROOT / "test_data.csv"
```

`Path(__file__).parent` is absolute — it resolves relative to the **script's** location, not
the shell's working directory. Streamlit Cloud does not launch from a predictable cwd, so
`train_models.py`'s relative `Path("model")` style would break here. Different context,
different rule.

### Caching (lines 59–66)

```python
@st.cache_resource
def load_model(key):
    return joblib.load(MODEL_DIR / f"{key}.joblib")
```

Streamlit reruns the **entire script top-to-bottom on every widget interaction**. Without
caching, switching the dropdown would re-read a 3.5 MB KNN from disk each time.
`st.cache_resource` keeps one copy in memory across reruns, keyed by `key`. This is why
switching models in the UI feels instant.

### Upload handling (lines 113–128)

```python
with st.sidebar:
    model_label = st.selectbox("Classifier", list(MODEL_CHOICES))
    upload = st.file_uploader("Evaluation CSV", type=["csv"])

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

### Validation and the column-order guarantee (lines 130–142)

```python
missing = [column for column in FEATURES if column not in frame.columns]
if missing:
    st.error(f"Uploaded file is missing {len(missing)} required column(s): ...")
    st.stop()

model = load_model(MODEL_CHOICES[model_label])
X = load_scaler().transform(frame[FEATURES])
```

Two things happen in that last line, both essential:

1. **`frame[FEATURES]`** selects columns **by name, in training order** — so a CSV with the
   right columns in a scrambled order still scores correctly.
2. **`load_scaler().transform(...)`** applies the *persisted* training means and standard
   deviations. It does not fit a new scaler on the uploaded data. If it did, the same client
   would be scored differently depending on who else happened to be in the file.

`st.stop()` halts the script — without it, the subsequent `frame[FEATURES]` would raise a
`KeyError` and show a stack trace to the user.

### Scoring and rendering (lines 147–178)

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
the cell showing `TN = 4,402` would otherwise be dark text on a dark square.

`matplotlib.use("Agg")` (line 26) selects the non-interactive backend. On a headless
server there is no display to draw into; without `Agg`, plotting can fail at import time.
The `# noqa: E402` comment suppresses the linter's "import not at top of file" complaint,
because `matplotlib.use()` must run **before** `pyplot` is imported.

---

## 4.5 Data flow, end to end

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
      │  tune → clone → refit on 23,972│
      ▼                                ▼
5 fitted models ──────────────► score on test
      │                                │
      │  joblib.dump                   ▼
      ▼                        reports/metrics.json
model/*.joblib                 reports/details.json
model/scaler.joblib
model/metadata.json
      │
      ▼
app.py  ── loads artefacts, transforms uploads with the saved scaler, predicts
```

## 4.6 Running it

```bash
python prepare_data.py      # ~2 s   → data/credit_default.csv
python train_models.py      # ~60 s  → model/, reports/, test_data.csv
streamlit run app.py        # opens http://localhost:8501
```

`train_models.py` is dominated by the SVM: fitting 23,972 points with an RBF kernel took
**55,422 iterations** and produces 10,506 support vectors.

Next: [05_why_these_results.md](05_why_these_results.md) ties every number back to a
mechanism.
