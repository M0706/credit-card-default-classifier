# 1. Theory — What the Five Algorithms Are Actually Doing

## 1.1 The problem setting

Supervised binary classification. We have a dataset of pairs $(x_i, y_i)$ where
$x_i \in \mathbb{R}^{23}$ is a client's feature vector and $y_i \in \{0, 1\}$ is
whether they defaulted.

We want a function $f: \mathbb{R}^{23} \rightarrow \{0,1\}$ such that $f(x)$ predicts
the label of clients **we have never seen**. This distinction is everything:

> We are not trying to fit the 23,972 training rows well. We are trying to perform well
> on the 5,993 rows we deliberately hid.

Fitting the training rows well is trivial — memorise them. The whole discipline of
machine learning is about generalising past them.

### Why the train/test split exists

If we scored a model on the same data it learned from, a lookup table would score
100%. So we split **before** doing anything else:

| Partition | Rows | Purpose |
|---|---|---|
| Training | 23,972 (80% of 29,965) | Models learn from these |
| Test | 5,993 (20%) | Sealed. Scored once, at the end |

`random_state=42` fixes the split, so it is identical on every machine.

### Why the split is *stratified*

We called `train_test_split(..., stratify=y)`. This forces both partitions to have the
same 22.1% default rate. Without it, a random split could hand us a test set that is,
say, 18% positive — and every metric would shift for reasons that have nothing to do
with model quality.

### The third partition you'll see in the code

`train_models.py` makes a *second* split, inside the training data:

```
29,965 total
└── 23,972 training
    ├── 19,177  "fit" split   ← candidate models are fitted here during tuning
    └── 4,795   validation    ← candidate models are compared here
└── 5,993 test                 ← sealed until final scoring
```

**Why:** to choose the decision tree's depth we need to compare candidates. If we
compared them on the test set, we would be tuning on the answer key and our final
score would be optimistic. The validation split is a stand-in question set.

**Why the winners are then refit on all 23,972:** once the depth is chosen, the
4,795 validation rows are wasted if we leave them aside. Throwing them back in gives the
final model 25% more data. This refit is what changed our Decision Tree's MCC from
0.3858 to **0.3912**. (The first build of this project forgot the refit — the model was
scored while still trained on only 19,177 rows. See `04_code_walkthrough.md` §4.3.)

---

## 1.2 The bias–variance idea, with our own evidence

Every model errs for one of two reasons:

- **Bias** — the model is too simple to represent the true relationship (underfitting).
- **Variance** — the model is so flexible it fits noise that won't repeat (overfitting).

You can watch this trade-off in our tuning output:

| `max_depth` | Validation F1 |
|---|---|
| 4 | 0.4605 |
| 6 | **0.4609** ← chosen |
| 8 | 0.4432 |
| 10 | 0.4323 |
| unlimited (`None`) | 0.4323 |

Accuracy gets *worse* once the tree goes past depth 6. The unlimited tree can perfectly
memorise the training data — its training accuracy is 1.000 — yet it generalises worse
than a 44-leaf tree. Each extra split is fitting quirks of those specific 19,177
people that don't describe the other 5,993.

This table is the single most persuasive piece of evidence you can bring to a viva that
you understand overfitting: **more capacity made the model worse, measurably.**

---

## 1.3 Logistic Regression

### The model

A linear model in the log-odds:

$$z = w \cdot x + b \qquad p = \sigma(z) = \frac{1}{1 + e^{-z}} \qquad
\log\frac{p}{1-p} = w \cdot x + b$$

The sigmoid $\sigma$ squashes any real number into $(0,1)$, so the output is a valid
probability. Solving for $p$ given the log-odds is exactly what $\sigma$ inverts.

**Decision rule:** predict `default` when $p \ge 0.5$, i.e. when $w \cdot x + b \ge 0$.
This is a **hyperplane** — a flat 22-dimensional surface — so the boundary between
classes is linear.

### How it is trained

Minimise the binary cross-entropy (log loss):

$$L(w,b) = -\frac{1}{N}\sum_i \Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big]$$

This is the negative log-likelihood of the data under the model. There is no closed
form, so it is solved iteratively. Our implementation (`models/logistic.py`) uses
**Newton's method** — IRLS — which needs the 24×24 Hessian but converges in **7
iterations**; scikit-learn's reference version uses **L-BFGS**, a quasi-Newton
optimiser that avoids forming the Hessian and scales to many more features. On this
problem both reach the same optimum (`validate_models.py`: AUC gap 0.0000).

### Reading our actual coefficients

Standardised features mean one unit = one standard deviation.

| Feature | Coefficient | Odds multiplier ($e^{w}$) |
|---|---|---|
| `PAY_0` | **+0.6492** | **×1.91** |
| `BILL_AMT1` | −0.4605 | ×0.63 |
| `BILL_AMT2` | +0.2832 | ×1.33 |
| `PAY_AMT1` | −0.2584 | ×0.77 |
| `PAY_AMT2` | −0.1915 | ×0.83 |
| `LIMIT_BAL` | −0.1052 | ×0.90 |

Read `PAY_0 = +0.6492` as: *holding everything else fixed, one standard deviation more
delinquency in the most recent month multiplies the odds of default by 1.91.* That is
by far the strongest single signal — which is precisely why the decision tree split on
it first.

The intercept is **−1.4687**, so a client at exactly the mean on every feature has
log-odds −1.4687, i.e. probability $\sigma(-1.4687) = 0.187$ — close to the 22.1% base
rate, as it should be.

### Why its recall is only 0.25

Logistic regression minimises log loss, and under a 78/22 imbalance it is *rational*
for the model to be conservative: predicting 0.5+ for a client means committing to an
event that happens only 22% of the time. It therefore only raises the flag when the
evidence is overwhelming.

### Strengths and weaknesses

- ✅ Fast, stable, coefficients are interpretable as odds ratios.
- ✅ Genuinely calibrated probabilities (unlike NB or SVM).
- ❌ Cannot represent interactions, e.g. "high bill **and** rising delinquency is much
  worse than either alone". It would need those as explicit products.

---

## 1.4 Decision Tree

### The model

Recursively partition the feature space with axis-aligned splits until each region is
pure. Each internal node asks one question of the form `PAY_0 <= 1.34`.

Our fitted tree, printed by `export_text`:

```
|--- PAY_0 <= 1.34
|   |--- PAY_2 <= 1.36
|   |   |--- PAY_AMT3 <= -0.25
|   |   |   |--- ...
|   |--- PAY_2 >  1.36
|   |   |--- LIMIT_BAL <= -0.71
|   |   |   |--- ...
|--- PAY_0 >  1.34
|   |--- BILL_AMT1 <= -0.69
|   |   |--- class: 0
|   |--- BILL_AMT1 >  -0.69
|   |   |--- PAY_6 <= 1.12
|   |   |   |--- ...
```

Note the thresholds are in **standardised** units — we scaled before fitting, and
scaling is monotonic so it doesn't change the tree's structure, only how the cut points
print. `PAY_0 <= 1.34` means "at most 1.34 SD above the mean".

### How the split is chosen

At each node, consider every feature and every candidate threshold, and pick the one
that most reduces **Gini impurity**:

$$\text{Gini}(t) = 1 - \sum_{k \in \{0,1\}} p_k^2$$

A pure node has Gini 0; a 50/50 node has Gini 0.5. The gain from a split is

$$\Delta I = I(\text{parent}) - \frac{n_L}{n}I(\text{left}) - \frac{n_R}{n}I(\text{right})$$

The algorithm is **greedy**: it takes the best split *now* with no lookahead, then
recurses. That's why it's fast, and also why it isn't guaranteed to find the globally
best tree.

### Feature importance, from our actual tree

`DecisionTreeModel.feature_importances_` is the normalised total impurity decrease
attributable to each feature:

| Feature | Importance |
|---|---|
| `PAY_0` | **0.6665** |
| `PAY_2` | 0.1383 |
| `PAY_AMT3` | 0.0536 |
| `PAY_3` | 0.0373 |
| `LIMIT_BAL` | 0.0296 |
| `BILL_AMT1` | 0.0185 |

**Two-thirds of the model's entire decision-making is one column.** The rest is refinement.
If you had to build a one-question credit screen, it would be "how late was your last
payment?" — and the tree quantifies exactly how much that's worth.

Our settings: `max_depth=6`, `min_samples_leaf=50`, which produces 44 leaves. The
leaf-size floor prevents splits that isolate 3 people, which is where memorisation starts.

### Strengths and weaknesses

- ✅ Fully interpretable — you can hand a path to a loan officer and justify it.
- ✅ Needs no scaling (it only compares values against thresholds).
- ✅ Captures non-linearities and interactions automatically.
- ❌ High variance: change 5% of the data and the structure can change visibly.
- ❌ Axis-aligned only — a diagonal boundary must be approximated by a staircase.

---

## 1.5 K-Nearest Neighbours

### The model

There is no training phase. `fit()` just stores the data. To predict, find the $k$
closest training points to the query and let them vote.

$$d(x, x') = \sqrt{\sum_j (x_j - x'_j)^2}\qquad
\hat{y} = \arg\max_k \sum_{i \in N_k(x)} w_i \,\mathbb{1}[y_i = k]$$

We used `n_neighbors=15`, `weights="distance"`, so $w_i = 1/d(x, x_i)$ — closer
neighbours count more. Distance weighting matters when a small `k` means one very close
point could otherwise be outvoted.

### Why scaling is non-negotiable here

Euclidean distance sums squared differences across all 23 features. `LIMIT_BAL` ranges
10,000–1,000,000 while `AGE` ranges 21–79. Unscaled, a 200,000 difference in credit
limit contributes $(200{,}000)^2 = 4\times10^{10}$ while a 40-year age difference
contributes only $1{,}600$. **Age would be invisible.** After standardisation both
contribute comparably.

### Why `k=15` won

| `k` | Validation F1 |
|---|---|
| 15 | **0.4365** |
| 25 | 0.4218 |
| 45 | 0.4163 |
| 75 | 0.3940 |

Smaller `k` = more local = more variance. Larger `k` = smoother = blurs the minority
class away, because with only 22% positives, a neighbourhood of 75 is usually ~58
non-defaulters, so the vote is nearly always "no default". The F1 decline with `k` is a
direct measurement of that dilution.

### The curse of dimensionality

In 23 dimensions, points spread out and distances **concentrate**: the ratio between the
nearest and farthest neighbour approaches 1. "Nearest" stops meaning "similar". This is
why KNN ranks mid-table despite being conceptually simple, and why its 3.5 MB artefact
(23,972 stored points) is the least deployable of the five.

### Strengths and weaknesses

- ✅ Zero assumptions about the shape of the boundary.
- ✅ Naturally multi-class.
- ❌ Prediction cost scales linearly with dataset size.
- ❌ Degrades in high dimensions.
- ❌ Sensitive to `k`, and to irrelevant features (which still contribute distance).

---

## 1.6 Gaussian Naive Bayes

### The model

Bayes' theorem, plus one very strong assumption:

$$P(y \mid x) = \frac{P(y)\,P(x \mid y)}{P(x)}
\;\;\propto\;\; P(y)\prod_{i=1}^{23} P(x_i \mid y)$$

The simplification is $\;P(x \mid y) = \prod_i P(x_i \mid y)\;$ — this says the 23
features are **conditionally independent given the class**. That is almost never true,
hence "naive".

Each factor is modelled as a Gaussian:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{iy}^2}}
\exp\!\left(-\frac{(x_i - \mu_{iy})^2}{2\sigma_{iy}^2}\right)$$

Training is just computing a mean $\mu$ and variance $\sigma^2$ per feature per class —
hence the 1.5 KB artifact and near-instant fit.

### The numbers our fit produced

| Quantity | Class 0 (no default) | Class 1 (default) |
|---|---|---|
| Prior $P(y)$ | 0.7787 | 0.2213 |
| Training count | 18,668 | 5,304 |

Priors are just the training base rates.

### Why its recall is 0.67 and its MCC is 0.31 — the interesting part

`BILL_AMT1` … `BILL_AMT6` are **six measurements of nearly the same thing**. Under the
independence assumption, the model treats them as six *independent* pieces of evidence.
If a client has a high bill, all six are high, and Naive Bayes multiplies that evidence
six times over. The posterior is pushed wildly toward class 1.

The same happens with `PAY_AMT1`…`PAY_AMT6`. The result is a model that:

- catches **889 of 1,326** defaulters (recall 0.6704 — highest of all five),
- but raises **1,412 false alarms** (precision 0.3864 — lowest of all five).

Its confusion matrix has 1,412 false positives versus 265 for the Decision Tree. In a
bank, that means 1,412 good customers incorrectly penalised. **Its high recall is not
skill — it is a systematic bias toward predicting the minority class.**

This is exactly why MCC exists: NB's recall looks great in isolation, but MCC (0.3140)
correctly identifies it as the weakest of the five.

### Strengths and weaknesses

- ✅ Extremely fast, tiny artifact, works with very little data.
- ✅ Good for high-dimensional sparse data (text — Multinomial NB).
- ❌ Probabilities are badly miscalibrated; do not price risk off them.
- ❌ Correlated features break it in a predictable, one-directional way.

---

## 1.7 Support Vector Machine

### The model

Find the hyperplane that separates the classes with the **widest margin**. Among all
separating planes, the maximum-margin one is the most robust to new points near the
boundary.

Soft-margin formulation:

$$\min_{w,b,\xi}\;\; \frac{1}{2}\lVert w\rVert^2 + C\sum_i \xi_i
\quad\text{subject to}\quad y_i(w \cdot x_i + b) \ge 1 - \xi_i,\;\; \xi_i \ge 0$$

- $\frac12\lVert w\rVert^2$ — maximise the margin (margin $= 2/\lVert w\rVert$).
- $C\sum\xi_i$ — penalise violations. **`C` is the dial:** small `C` = wide margin,
  more violations tolerated (more regularised); large `C` = narrow margin, fewer
  violations (more overfitting risk). We used the default `C=1.0`.

### The kernel trick

Our boundary is non-linear because we replaced the dot product with an RBF kernel:

$$K(x, x') = \exp(-\gamma \lVert x - x'\rVert^2), \qquad
\gamma_{\text{scale}} = \frac{1}{n_{\text{features}}\cdot \mathrm{Var}(X)}$$

This implicitly maps the 23 features into an infinite-dimensional space where a linear
separator exists, **without ever computing that mapping**. Prediction becomes:

$$\hat{y} = \operatorname{sign}\!\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)$$

Only points with $\alpha_i > 0$ — the **support vectors** — contribute. From our fit:

> **1,877 of the 4,000** subsampled training points became support vectors (47%).

That's a high proportion, which is typical of an overlapping, noisy boundary like credit
default. It also explains why the artifact is 351 KB — modest, because the training set
itself is capped at 4,000 rows — while prediction still evaluates the kernel against
1,877 points for every client. It is the most expensive model to serve.

### Why scaling is mandatory

The RBF kernel is a function of Euclidean distance. As with KNN, unscaled features let
`LIMIT_BAL` swamp everything, and $\gamma$ (which multiplies distance) becomes
meaningless. Standardisation is not optional here.

### Why we score AUC with `decision_function`

`SVC(probability=False)` has no `predict_proba`. Our code does:

```python
scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)
```

`decision_function` returns the signed distance to the boundary. AUC only needs a
*ranking* of clients by risk, and the signed distance is a valid ranking — so this
avoids the slow internal Platt-calibration step, without changing AUC.

### Strengths and weaknesses

- ✅ Strong on non-linear, moderate-sized problems; robust in high dimensions.
- ✅ Memory-efficient *in training* (only support vectors matter).
- ❌ `C` and `gamma` need tuning; here left at defaults, so there is headroom.
- ❌ Slow to predict and to scale beyond ~100k rows.

---

## 1.8 Why one model doesn't win everywhere

| Model | Boundary shape | Needs scaling | Captures interactions | Interpretable | Trains on |
|---|---|---|---|---|---|
| Logistic Regression | Linear hyperplane | ✅ yes | ❌ no | ✅ coefficients | all rows equally |
| Decision Tree | Axis-aligned boxes | ❌ no | ✅ yes | ✅ rules | all rows equally |
| KNN | Arbitrary, local | ✅ yes | ✅ implicitly | ❌ no | stores everything |
| Gaussian NB | Quadratic (per-class Gaussian) | ❌ no | ❌ no (assumes independence) | ⚠️ partly | only means/variances |
| SVM (RBF) | Smooth non-linear | ✅ yes | ✅ via kernel | ❌ no | only support vectors |

No single row dominates every column — which is why the assignment asks you to compare
five rather than pick one. On this dataset, the tree's ability to express "if
`PAY_0 > 1.34` and `BILL_AMT1 > -0.69` then risk" is what beats the linear models'
inability to express that conjunction.

Next: [02_metrics.md](02_metrics.md) turns these predictions into honest numbers.
