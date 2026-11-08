# 5. Why These Results — Mechanism by Mechanism

This document ties every number in the results table to a property of the data or the
algorithm. If you can explain the content here, you can defend the project.

All figures are on the 5,993 held-out clients: **1,326 defaulters, 4,667 non-defaulters,
majority-class baseline 0.7787**.

---

## 5.1 First: where is the signal?

Before comparing models, it's worth knowing how predictable this problem is at all. Look
at the default rate by `PAY_0` (most recent repayment status), computed on the cleaned
dataset:

| `PAY_0` | Meaning | Clients | Default rate |
|---|---|---|---|
| −2 | No consumption | 2,750 | 13.2% |
| −1 | Paid in full | 5,682 | 16.8% |
| 0 | Revolving credit | 14,737 | 12.8% |
| **1** | 1 month late | 3,667 | **34.0%** |
| **2** | 2 months late | 2,666 | **69.1%** |
| **3** | 3 months late | 322 | **75.8%** |

Collapsing that into the tree's first split (`PAY_0 >= 2`, i.e. at least two months
delinquent):

| Group | Share of clients | Default rate |
|---|---|---|
| `PAY_0 < 2` | 89.6% | **16.6%** |
| `PAY_0 >= 2` | 10.4% | **69.5%** |

**A single yes/no question separates the population by more than fourfold.** That is why
the Decision Tree put its first split on `PAY_0` and why `PAY_0` accounts for **73.5%** of
the tree's total impurity reduction.

It also explains the ceiling on every model's score. Within the 89.6% low-risk group the
default rate is still 16.6% — thousands of clients who look identical on repayment
behaviour but default anyway. That residual is driven by information this dataset does
not contain (income shocks, job loss, other lenders' exposure). **No algorithm can recover
signal that isn't in the features.**

---

## 5.2 Reading the decision tree's rules in real units

The tree's thresholds print in standardised units. Converting back with
`raw = mean + z × std`:

| Split | z threshold | Mean | Std | Raw threshold |
|---|---|---|---|---|
| `PAY_0 <= 1.34` | 1.34 | −0.02 | 1.12 | **≈ 1.5 months** |
| `PAY_2 <= 1.36` | 1.36 | −0.12 | 1.19 | **≈ 1.5 months** |
| `BILL_AMT1 > −0.69` | −0.69 | 51,338 | 73,595 | **≈ NT$557** |
| `PAY_AMT3 > −0.25` | −0.25 | 5,249 | 18,060 | **≈ NT$734** |
| `LIMIT_BAL > −0.71` | −0.71 | 167,182 | 129,997 | **≈ NT$74,884** |
| `PAY_6 <= 1.12` | 1.12 | −0.26 | 1.13 | **≈ 1 month** |

So the learned rule, in business language, is roughly:

> **If the client is 2+ months delinquent *and* their last statement exceeded about
> NT$557, the model raises its risk estimate — and it tempers that if the client was
> delinquent six months ago too, because chronic lateness looks different from a sudden
> one-off.**

That last part is genuinely non-obvious and is the kind of interaction a linear model
cannot express.

---

## 5.3 Confusion matrices, side by side

| Model | TN | FP | FN | TP | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 4,543 | **124** | 995 | 331 | 0.7275 | 0.2496 | 0.9734 |
| Decision Tree | 4,402 | 265 | 836 | 490 | 0.6490 | 0.3695 | 0.9432 |
| KNN | 4,390 | 277 | 878 | 448 | 0.6179 | 0.3379 | 0.9406 |
| Naive Bayes | 3,255 | **1,412** | **437** | **889** | 0.3864 | **0.6704** | 0.6975 |
| SVM | 4,452 | 215 | 887 | 439 | 0.6713 | 0.3311 | 0.9539 |

Read the FP and FN columns as two different business costs:

- **FP = 124** (Logistic Regression) means 124 good customers wrongly flagged. Excellent
  for customer relations, terrible for catching fraud.
- **FN = 995** (Logistic Regression) means 995 defaulters sailed through unflagged. Each
  one is a written-off balance.

There is no configuration that minimises both. **This table *is* the precision–recall
trade-off**, made concrete in NT dollars.

---

## 5.4 Model by model

### Logistic Regression — confident and cautious

**Predicted behaviour:** smooth linear boundary, effectively no interactions, optimises
log loss under imbalance → conservative.

**Observed:** fewest false positives (124), lowest recall (0.2496), **highest precision
(0.7275)**. AUC 0.7160.

**Why the AUC is the lowest but not by much:** the true boundary really is largely
monotonic in `PAY_0` — late payers default more — and that is exactly what a linear model
captures. It fails only where the relationship is conditional (a high bill matters more
when delinquency is also rising). Those interactions are a minority of cases, so LR loses
only ~0.018 AUC to the tree.

**The calibration tell:** its precision is high and its recall low, but its AUC (0.7160)
is noticeably better than its F1 (0.3717) would suggest. That gap means the *ranking* is
decent while the *threshold* at 0.5 is badly chosen for this class balance — the model has
the information but the default cut-off discards it. §2.8 of `02_metrics.md` shows the
same effect for the tree.

---

### Decision Tree — the winner, for now

**Predicted behaviour:** captures `PAY_0` first (highest impurity gain), adds conditional
refinements, shallow depth stops it memorising.

**Observed:** best on all three headlining metrics — accuracy 0.8163, AUC 0.7341, MCC
0.3912 — with 15 leaves at depth 4.

**Why `PAY_0` dominates feature importance (73.5%):** impurity gain is computed at the
root, where the population is the full 50/50-ish mix; a split that separates 69.5%
defaulters from 16.6% defaulters produces an enormous Gini drop. Once that split exists,
every later split operates inside much purer subpopulations and therefore has less room to
improve — mechanically, downstream features *cannot* accumulate comparable importance.
**This is a known bias of impurity-based importance:** it favours features that split
early, and it is unreliable when features are correlated (the tree may split on `PAY_0`
and never give `PAY_2` its due, even though `PAY_2` carries almost the same information).

**The 13 → 15 leaf change** between the first build and the fix is the signature of the
refit bug in `04_code_walkthrough.md` §4.3: more training data produced a slightly richer
tree, which is exactly what you'd expect.

---

### KNN — decent ranking, expensive to serve

**Predicted behaviour:** local boundaries, needs `k` tuned, harmed by the curse of
dimensionality.

**Observed:** AUC 0.7327 (essentially tied with the tree's 0.7341), MCC 0.3545.

**Why its recall is middling (0.3379):** with distance weighting and `k=15`, a query near
the boundary is outvoted by the ~11 non-defaulters typically among its 15 neighbours. This
is the same minority-dilution effect that made `k=75` perform worst in tuning.

**Why its AUC is nevertheless near the top:** KNN is a pure ranking method — it never
assumes a functional form. If two clients have similar `PAY_0`, similar bills and similar
payments, KNN gets it right regardless of how those variables interact. That flexibility
shows up in AUC while the fixed 0.5 threshold still under-fires.

**The practical objection:** the artifact is 3.5 MB and every prediction is a 23,972-point
distance computation. The tree answers the same question from a 3 KB file in constant
time. Equal accuracy with 1,000× the serving cost is not a tie in practice.

---

### Naive Bayes — high recall, low skill

**Predicted behaviour:** independence assumption violated by six near-duplicate bill
columns → over-counts evidence → over-predicts the minority class.

**Observed:** recall **0.6704** (best), precision **0.3864** (worst), accuracy **0.6915**
(below the 0.7787 baseline!), MCC **0.3140** (worst).

**The decisive number:** accuracy 0.6915 is *worse than predicting "no default" for
everyone*. A model that is beaten by a constant deserves to be called out, and this is the
clearest example in the project of why accuracy alone is useless.

**Why the recall is high but hollow:** NB predicted default 2,301 times out of 5,993 —
38% of clients, versus a true rate of 22%. It is not detecting defaulters; it is
**over-flagging**. Of the 1,326 real defaulters it caught 889, but it also flagged 1,412
people who paid on time. Precision 0.3864 means **less than two in five of its alarms are
real**.

**Why the priors don't save it:** with priors 0.7787 / 0.2213 the model starts biased
*against* class 1 (log prior ratio ≈ −1.26). The correlated-evidence inflation overwhelms
that penalty. This is the crispest illustration in the whole project of what "naive"
costs.

---

### SVM — best boundary, worst serving cost

**Predicted behaviour:** non-linear RBF boundary, good generalisation, expensive
prediction.

**Observed:** accuracy 0.8161 (essentially tied with the tree's 0.8163 — a difference of
one client in 5,993), MCC 0.3794, AUC 0.7173.

**Why 10,506 of 23,972 points became support vectors (44%):** in a well-separated problem
only a handful of points sit on the margin. Here 44% do, which is a direct measurement of
how much the classes overlap — most clients are genuinely ambiguous. That's the same fact
the `PAY_0` table in §5.1 shows from a different angle.

**Why its AUC (0.7173) trails the tree's (0.7341) despite similar accuracy:** `C=1.0` and
`gamma='scale'` are the defaults, never tuned. A coarse grid over `C ∈ {0.1, 1, 10}` and
`gamma ∈ {0.01, 0.1, 1}` would almost certainly close that gap. We left it untuned to keep
the five-model comparison to a common, defensible procedure — but the headroom is real and
worth saying out loud.

**Why `decision_function` rather than probabilities:** see `01_theory.md` §1.7. It gives
the same AUC without Platt scaling.

---

## 5.5 Why all five cluster around 0.81 accuracy

| Model | Accuracy | vs. 0.7787 baseline |
|---|---|---|
| SVM | 0.8161 | +0.0374 |
| Decision Tree | 0.8163 | +0.0376 |
| Logistic Regression | 0.8133 | +0.0346 |
| KNN | 0.8073 | +0.0286 |
| Naive Bayes | 0.6915 | **−0.0872** |

Four models land within **0.9 percentage points of each other**. That is not a coincidence
and it is not a bug — it is the expected outcome when:

1. The positive class is 22% of the data, so ~78% of every prediction is "no default" and
   is right for free.
2. The signal is concentrated in one column (`PAY_0`). Every algorithm finds the same
   dominant feature; they differ only in how they refine around it.

**This is why the assignment asks for six metrics.** With accuracy alone, SVM edges out
Logistic Regression by 0.0028 and you would conclude "SVM is better". On MCC the gap is
0.0298, which is a far more meaningful separation.

---

## 5.6 But is the Decision Tree *really* the winner?

Bootstrap resampling — 300 resamples of the 5,993 test rows with replacement — gives
sampling intervals:

| Model | AUC [95% CI] | MCC [95% CI] |
|---|---|---|
| Logistic Regression | 0.7151 [0.6997, 0.7324] | 0.3496 [0.3200, 0.3748] |
| **Decision Tree** | **0.7340 [0.7179, 0.7485]** | **0.3905 [0.3602, 0.4178]** |
| KNN | 0.7325 [0.7172, 0.7494] | 0.3544 [0.3267, 0.3827] |
| Naive Bayes | 0.7276 [0.7102, 0.7435] | 0.3136 [0.2874, 0.3384] |
| SVM | 0.7176 [0.7032, 0.7328] | 0.3793 [0.3508, 0.4109] |

Read this honestly:

- **All five AUC intervals overlap.** The apparent ranking 0.7341 > 0.7327 > 0.7283 >
  0.7173 > 0.7160 is not statistically resolvable at this sample size. You cannot claim
  the tree ranks better than KNN; the difference (0.0014) is far inside the noise.
- **On MCC, the Decision Tree and Naive Bayes intervals do not overlap** (0.360–0.418 vs
  0.287–0.338). That one comparison *is* defensible: the tree is genuinely more skilful
  than Gaussian NB.
- **Tree vs SVM and tree vs KNN overlap** on MCC too. Those are ties.

So the accurate statement is:

> The Decision Tree has the best point estimate on every headline metric and is clearly
> superior to Naive Bayes, but it is statistically indistinguishable from SVM and KNN on
> this test set. Its decisive practical advantages are interpretability and a 3 KB
> artifact rather than a measurable accuracy lead.

That is a much stronger claim than "the tree wins", because it survives scrutiny. Saying
precisely what your evidence supports is worth more in a viva than an overconfident
ranking.

---

## 5.7 What would actually improve these numbers, in order

1. **Tune the operating threshold** (§2.8). Free. Threshold 0.40 lifts tree MCC from
   0.3912 to 0.3974 and F1 from 0.4709 to 0.4915 with no retraining. This is the highest
   value-per-line-of-code change available.
2. **`class_weight="balanced"`** on LR/SVM/tree. Reweights the loss so the minority class
   is no longer systematically under-predicted; typically doubles recall at a precision
   cost.
3. **Tune SVM `C` and `gamma`.** Currently at defaults, and it's already competitive. A
   9-point grid is cheap and likely closes the AUC gap.
4. **Add engineered features.** `mean(PAY_0…PAY_6)` (average lateness),
   `BILL_AMT1 / LIMIT_BAL` (utilisation ratio), `PAY_AMT1 / BILL_AMT1` (repayment fraction).
   Utilisation is a classic credit-risk feature and is not directly present.
5. **One-hot encode `PAY_*` as ordered categories** instead of treating −2/−1/0/1…8 as a
   continuous scale. Removes an assumption the data doesn't support.
6. **Gradient boosting (XGBoost/LightGBM).** Out of scope for this assignment's five-model
   list, but it is what would actually win on this dataset — typically AUC ≈ 0.78.

## 5.8 Honest limitations

- **Recall 0.37 is operationally weak.** Missing 63% of defaulters would not be acceptable
  in production. The project measures this rather than hiding it.
- **The `PAY_*` columns are treated as continuous.** −2 ("no consumption") is not one unit
  below −1 ("paid in full"); it is a different state entirely.
- **SVM and LR hyperparameters are untuned**, so the comparison is slightly unfair to them.
- **Single 80/20 split.** Cross-validation would give tighter estimates, though the
  bootstrap in §5.6 partly compensates.
- **No `random_state` dependence check.** All results use seed 42; a different seed would
  move the numbers a little. The bootstrap intervals indicate how much.
- **`feature_importances_` is impurity-based** and therefore biased toward
  high-cardinality, early-splitting features and unreliable under correlation. Permutation
  importance would be the more robust measure.
