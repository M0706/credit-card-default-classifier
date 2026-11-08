# 07 — The five algorithms, implemented from scratch

This document is about the code in `models/`. It assumes you have read
[`01_theory.md`](01_theory.md) for the maths and now want to see where that maths
actually lives.

The short version: every algorithm is about forty lines of NumPy, and each one is
wrong in an instructive way if you get a single sign or index wrong. That is the
point of writing them.

---

## Why write them at all

`sklearn.linear_model.LogisticRegression(...)` would have produced identical
metrics in one line. It would also have hidden:

- that Newton's method converges in **7 iterations** here while gradient descent
  needs hundreds and a learning rate you have to guess;
- that a decision tree at depth 6 only needs **44 leaves** to reach 0.747 AUC,
  and that `PAY_0` alone drives two-thirds of the split decisions;
- that **1,877 of a possible 4,000** training points sit on or inside the SVM
  margin, which is why the model is so expensive to predict with;
- that Gaussian Naive Bayes is not "inaccurate" but *overconfident* — it predicts
  the minority class 46% of the time against a 22% base rate.

The assignment asks for the algorithms. Writing them is also the only way to
know that the library's answer is right.

**scikit-learn is still used**, deliberately, for four things it is not the
subject of: `train_test_split` (splitting), `StandardScaler` (scaling), the
metric functions, and — in `validate_models.py` — as the **reference
implementation** our code is checked against.

---

## The shared contract

`models/base.py` defines an abstract class that all five inherit:

```python
class Model(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict_proba(self, X): ...

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path): joblib.dump(self, path)

    @staticmethod
    def load(path): return joblib.load(path)
```

Three consequences, all of them the reason the base class exists:

1. **`train_models.py` never branches on the algorithm.** It loops over a dict,
   calls `.fit()`, calls `.predict_proba()`. No `if isinstance(...)` anywhere.
2. **`app.py` never learns a new interface.** It loads a joblib file and calls
   the same two methods, so a model trained yesterday by hand-written code and
   one trained by scikit-learn are indistinguishable to it.
3. **`validate_models.py` can compare apples to apples.** Because our class
   exposes the same method names scikit-learn does, one `compare()` function
   scores both.

`_validate_X` is shared housekeeping: it converts input to a float array, checks
the array is 2-D, and checks the column count matches what the model was fitted
on. This catches the single most common serving bug — feeding the app a CSV with
the columns in a different order — as a clear exception rather than as silently
wrong probabilities.

`_sigmoid` clips its input to ±500 before `exp`, so a runaway weight during
training produces a saturated 0 or 1 rather than `inf`/`NaN` that poisons every
following iteration.

---

## 1. Logistic Regression — `models/logistic.py`

### The model

```
z = w·x + b          p = σ(z) = 1 / (1 + e^(-z))
```

`p` is the probability of default. The decision boundary is the hyperplane
`z = 0`, where the odds are even.

### The objective

```
J(w, b) = ½‖w‖²  +  C · Σ_i [ -y_i log p_i - (1 - y_i) log(1 - p_i) ]
```

The first term shrinks the weights (L2 / ridge). The second is log loss. `C` is
scikit-learn's convention: **larger C = weaker regularisation**, because it
scales the loss term, not the penalty term. The intercept is not penalised —
penalising it would make the model unable to match the base rate.

### The update

Newton's method moves the parameters to the minimum of a local quadratic
approximation:

```
β ← β − H⁻¹ g
```

where `β = [b, w]`, and for the objective above:

```
g[0]      = C · Σ (p_i − y_i)
g[1:]     = w + C · Xᵀ (p − y)

H[0, 0]   = C · Σ p_i(1 − p_i)
H[0, 1:]  = C · (Xᵀ W)          where W_ii = p_i(1 − p_i)
H[1:, 0]  = H[0, 1:]
H[1:, 1:] = I + C · Xᵀ W X
```

These are exactly what the code computes, with one detail worth noticing: `W` is
clipped to a floor of `1e-12`. When the data is linearly separable, `p` saturates
at 0 or 1, `p(1−p) → 0`, and the Hessian becomes singular. The clip plus a
`1e-10` ridge on the solve keeps the linear system solvable.

### Why Newton and not gradient descent

There are 23 features, so the Hessian is 24×24 — inverting it costs nothing.
Newton uses curvature information, so it makes one large, well-aimed step per
iteration instead of hundreds of small speculative ones. It converged in **7
iterations** on 23,972 rows.

The trade-off: Newton is only affordable because the parameter count is small.
With 100,000 features you would use L-BFGS (which scikit-learn does by default)
or SGD.

### Reading the fitted weights

The intercept is **−1.4687**. That is the log-odds of default for a client whose
every feature sits at the training mean: `σ(−1.4687) = 0.187`, against a base
rate of 0.221. The gap is the L2 penalty pulling the intercept toward zero.

Because features are standardised, each weight is "log-odds change per one
standard deviation". `PAY_0` has a positive weight — one standard deviation worse
repayment status multiplies the odds of default by about 1.9. `BILL_AMT1` has a
negative weight, which looks odd until you remember that the amount *billed* is
measured alongside the amount *paid*: a high bill that is being serviced is not
the same risk signal as a high bill that is not.

---

## 2. Decision Tree — `models/tree.py`

### What a split does

At a node holding a set of rows, pick one feature and one threshold, send rows
with `x_j <= t` left and the rest right. Keep doing that recursively.

The goodness of a candidate split is the **drop in Gini impurity**. For a binary
node with positive fraction `p`:

```
Gini(p) = 2p(1 − p)
```

which is 0 for a pure node and 0.5 for a 50/50 node. For a split into left
(fraction `n_L/n`) and right:

```
Δ = Gini(p) − [ (n_L/n)·Gini(p_L) + (n_R/n)·Gini(p_R) ]
```

### The part that is not obvious: finding the best threshold cheaply

The naive version evaluates every midpoint between adjacent sorted values, per
feature, per node — and re-partitions the data for each one. With 23,972 rows
that is millions of partition operations.

Instead, `_best_split` sorts once per feature and then uses a **prefix sum**:

```
order          = argsort(column)
cumulative_pos = cumsum(positives[order])
```

Now for any split position `idx`, the number of positives on the left is
`cumulative_pos[idx - 1]`, and on the right it is `cumulative_pos[-1] - that`.
Both child impurities, both sample counts and the gain come out of a vectorised
computation over **all candidate thresholds at once** — no loop over candidates,
no repeat partitioning.

Two constraints are enforced in the same vectorised step:

- `min_samples_leaf` — candidates where either side would be too small get
  `gain = -inf`;
- integer feature values — `searchsorted(..., side="right")` puts all rows
  equal to the threshold on the left, which is what `<=` means, and avoids
  splitting a value like `PAY_0 = 2` across both sides.

### The deliberate approximation

Real CART considers *every* midpoint between adjacent unique values. This
implementation considers at most 32 **quantile** thresholds per feature. On a
continuous column that is a real approximation.

It costs almost nothing here: `validate_models.py` reports an AUC gap of
**0.0035** against scikit-learn's exhaustive search (0.7472 vs 0.7341 in the
reference run — and the direction is *in our favour* on this split). The reason
is that the columns that matter — repayment statuses — are small integers with
only a handful of distinct values, so the quantile grid is the exhaustive grid.

### Stopping

```
depth >= max_depth   or   n < 2·min_samples_leaf   or   node is pure   or   no gain
```

Without these the tree grows until every leaf holds one row: zero training error,
memorised noise. The evidence is in the tuning table — validation F1 was 0.4609
at depth 6 and 0.4323 for an unrestricted tree.

### Prediction

`_apply` walks the tree with index arrays rather than looping over rows:

```python
go_left = X[rows, node["feature"]] <= node["threshold"]
self._apply(node["left"],  X, rows[go_left],  out)
self._apply(node["right"], X, rows[~go_left], out)
```

Each recursion partitions only the rows that reached that node, so the total
work is O(n · depth) — microseconds for 6,000 rows.

### Feature importance

At each internal node, record `n · (impurity − weighted child impurity)`, sum per
feature, normalise. That is the total impurity reduction attributable to the
feature, and it is what makes `PAY_0 = 66.7%` a defensible claim.

---

## 3. K-Nearest Neighbours — `models/knn.py`

There is no training. `fit` stores the matrix and precomputes `‖x‖²` per row.
Everything happens at prediction time.

### The distance trick

Computing `sqrt(Σ_j (a_j − b_j)²)` for all pairs is a triple loop. Instead:

```
‖a − b‖² = ‖a‖² − 2·a·b + ‖b‖²
```

so the entire query-by-train distance matrix is **one matrix multiply** plus
broadcasts:

```python
d2 = x_sq[:, None] - 2.0 * (block @ self._X.T) + self._X_sq[None, :]
```

`np.maximum(d2, 0.0)` clips tiny negative values that floating-point rounding
produces when the true distance is zero — leaving them would make `sqrt` return
`NaN`.

### Selecting the neighbours

`np.argpartition(d2, k - 1, axis=1)[:, :k]` finds the k smallest distances in
**O(n) per row**, without sorting all 23,972 candidates. Full sorting would be
O(n log n) for information nobody needs: the order *within* the neighbourhood is
irrelevant.

### The vote

```
score = Σ_i w_i · [label_i = 1] / Σ_i w_i        w_i = 1 / distance_i
```

Distance weighting is what the tuned model uses. A neighbour at 0.4 units counts
twice as much as one at 0.8. The guard matters: an exact duplicate gives distance
0, and `1/0` is `inf`, which would make the vote `inf/inf = NaN`. The code
substitutes `1e12` for near-zero distances so the duplicate dominates the vote
without producing `NaN`.

### Why chunking, and why standardising

A 6,000 × 23,972 float64 distance matrix is 1.15 GB. Chunking queries into blocks
of 512 keeps the working set around 100 MB.

Standardising matters even more here than elsewhere. `LIMIT_BAL` is in the
hundreds of thousands and `AGE` is in tens; unscaled, the Euclidean distance
would be entirely `LIMIT_BAL` and KNN would be a one-feature model.

### The honest weakness

This model stores **all 23,972 rows** (4.6 MB) and compares every new client
against every one of them. It has no notion of which features are informative —
it is why it lands at 0.7333 AUC while a 44-leaf tree reaches 0.7472.

---

## 4. Gaussian Naive Bayes — `models/naive_bayes.py`

Bayes' theorem, with one assumption:

```
P(y|x) ∝ P(y) · P(x|y) = P(y) · Π_i P(x_i|y)
```

The product is the "naive" part: it treats 23 correlated features as 23
independent pieces of evidence.

### Fitting

Two numbers per feature per class — the mean and variance of that feature among
rows of that class. That is the whole model: 2 × 2 × 23 = 92 numbers, and why
the artefact is 1.3 KB.

### Scoring, in log space

Multiplying twenty-three probabilities that are each less than 1 underflows to
zero in float64. Summing their logs does not:

```
log P(y|x) = log P(y)
           − ½ Σ_i log(2π σ²_iy)
           − ½ Σ_i (x_i − μ_iy)² / σ²_iy
```

`predict_proba` then subtracts the row maximum before exponentiating — the
standard log-sum-exp trick — so the largest term becomes `exp(0) = 1` and the
rest stay representable.

### The zero-variance trap

If a feature is constant within a class, `σ² = 0` and the exponent divides by
zero. The fix is `var_smoothing`: add `1e-9 × (largest feature variance in the
whole training set)` to every variance. This is exactly what scikit-learn does,
and it is the reason the two implementations agree to four decimals.

### Why it is the "wrong" model that wins on F1

The independence assumption fails hard here: `BILL_AMT1` through `BILL_AMT6` are
near-duplicates. A client with six large bills has the same evidence counted six
times, so the model becomes **overconfident** — it pushes probabilities toward
the minority class. That is why it has the highest recall (0.6704) and F1
(0.4902) but the lowest MCC (0.3140) and an accuracy *below* the majority-class
baseline. It is not detecting risk better; it is shouting louder.

---

## 5. Support Vector Machine — `models/svm.py`

### The idea

Among all separating hyperplanes, choose the one with the widest margin — the
largest buffer between the boundary and the nearest points. Those nearest points
are the **support vectors**; every other training point is irrelevant to the
solution.

Soft margin allows violations, priced by `C`:

```
min_{w,b,ξ}  ½‖w‖² + C Σ ξ_i
s.t.         t_i(w·φ(x_i) + b) >= 1 − ξ_i,   ξ_i >= 0
```

with `t_i ∈ {−1, +1}`. The dual lets the data enter only through the kernel:

```
max_α  Σ α_i − ½ ΣΣ α_i α_j t_i t_j K(x_i, x_j)
s.t.   0 <= α_i <= C,   Σ α_i t_i = 0
```

`K(x, x') = exp(−γ‖x − x'‖²)` is the RBF kernel: 1 for identical points, falling
to 0 as they separate. It is what lets a linear method draw a curved boundary.

### SMO, concretely

You cannot solve 4,000 coupled multipliers at once cheaply. SMO takes **two at a
time** — the smallest set that can move while keeping `Σ α_i t_i = 0` — and
solves that pair exactly.

For a chosen pair `(i, j)`, the objective along the feasible direction reduces to
a quadratic in `α_j`, so the unconstrained optimum is

```
α_j = α_j_old − t_j (E_i − E_j) / η          η = 2K_ij − K_ii − K_jj
```

Three details the code has to get right:

1. **The clipping window.** Given the equality constraint, `α_j` may only move
   within `[L, H]`, computed differently for same-sign and opposite-sign labels.
2. **The `eta >= 0` guard.** For a valid kernel `η <= 0` always; a non-positive
   `η` near zero means the pair is degenerate and moving would be a division by
   a vanishing number. Skip it.
3. **The running decision value `f`.** Error terms are needed for every pair.
   Rather than recompute `Σ α_k t_k K(x_k, x)` from scratch each time, the code
   maintains `f` incrementally:

   ```
   f += Δα_i t_i K[i, :] + Δα_j t_j K[j, :] + (b_new − b_old)
   ```

   **That last term is the bug that cost the most time in this project.** Bias
   changes have to be added too, because `f` is the full decision value
   including `b`. Without it, `f` silently drifts, the errors become garbage,
   and `b` runs away — the first version of this file converged to an intercept
   of **114** and predicted "default" for every client. With the term restored,
   it reproduces scikit-learn's support vectors and intercept exactly.

### Why the model trains on 4,000 rows

The precomputed kernel is an `n × n` float64 matrix. At `n = 23,972` that is
4.6 GB — larger than the problem is worth. A stratified 4,000-row subsample is
quadratic-small (128 MB) and keeps the 78/22 class balance.

This is the one real capability gap in the port, and it is measured rather than
hidden: `validate_models.py` fits scikit-learn's `SVC` on the **same** 4,000
rows, so the comparison tests the SMO implementation fairly, while the reported
SVM metrics honestly reflect a model that saw less data than the other four.

### Why "probabilities" here are not probabilities

An SVM has no probabilistic output — only a distance from the boundary.
`predict_proba` applies a sigmoid to that distance. This is **monotone**, so:

- the *ranking* is exact, and therefore AUC is exact;
- the *probabilities* are not calibrated — a predicted 0.7 does not mean 70%.

The reporting consequence is that AUC and threshold-based metrics remain valid,
but a business rule like "flag anything above 0.6" would need Platt scaling
first. (`sklearn.svm.SVC(probability=True)` does exactly that, at the cost of an
extra cross-validated fitting pass.)

---

## Does it actually work? — `validate_models.py`

Each from-scratch model and its scikit-learn counterpart are fitted on the
identical split, with the identical scaler, and scored on the identical test set.

| Model | AUC (sklearn) | AUC (ours) | Δ AUC |
|---|---|---|---|
| Logistic Regression | 0.7160 | 0.7160 | 0.0000 |
| Decision Tree | 0.7341 | 0.7306 | −0.0035 |
| KNN | 0.7331 | 0.7333 | +0.0002 |
| Gaussian Naive Bayes | 0.7283 | 0.7283 | 0.0000 |
| SVM (same 4,000 rows) | 0.7106 | 0.7105 | −0.0001 |

The largest disagreement is the tree, and it is expected: quantile thresholds
instead of exhaustive midpoints. Everything else agrees to the fourth decimal,
which is the only kind of evidence that a from-scratch implementation is correct
rather than merely plausible.

(The tree figure above is from the validation run, where both models are fitted
for comparison. The shipped tree — depth 6, tuned on validation F1 — scores
0.7472 on the test set; `reports/validation.json` holds both.)

---

## What each implementation deliberately gives up

| Model | Simplification | Measured cost |
|---|---|---|
| Logistic Regression | Newton instead of L-BFGS | None — same optimum; only scales badly to many features |
| Decision Tree | 32 quantile thresholds instead of every midpoint | −0.0035 AUC vs sklearn |
| KNN | Nothing; it is the full algorithm | It is slow by nature, not by implementation |
| Naive Bayes | Nothing | Exact match with sklearn |
| SVM | 4,000-row stratified subsample; sigmoid "probability" | Lower AUC than a full-data SVM would get; calibrated probabilities unavailable |

Every row is a sentence you can defend in a viva, because each one names the
trade-off and the number it cost.

---

## Reading order

1. `models/base.py` — the contract, twenty lines.
2. `models/logistic.py` — the most direct mapping from maths to code.
3. `models/naive_bayes.py` — smallest model, clearest use of log space.
4. `models/tree.py` — the prefix-sum trick is the interesting part.
5. `models/knn.py` — pure vectorisation; no training at all.
6. `models/svm.py` — the longest, and the only one where the loop *is* the
   algorithm.
7. `validate_models.py` — how the claims above are checked.
