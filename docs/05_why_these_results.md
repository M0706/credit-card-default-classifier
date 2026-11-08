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
the Decision Tree put its first split on `PAY_0` and why `PAY_0` accounts for **66.7%** of
the tree's total impurity reduction, with `PAY_2` adding a further 13.8%.

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
| `PAY_0 <= 1.35` | 1.35 | 0.0 | 1.1 | **≈ 1.5 months** |
| `PAY_2 <= 1.36` | 1.36 | −0.1 | 1.2 | **≈ 1.5 months** |
| `PAY_AMT3 <= −0.27` | −0.27 | 5,224 | 17,711 | **≈ NT$466** |
| `BILL_AMT1 <= −0.69` | −0.69 | 51,271 | 73,241 | **≈ NT$883** |
| `LIMIT_BAL <= −1.09` | −1.09 | 167,273 | 129,867 | **≈ NT$26,364** |
| `AGE <= −0.43` | −0.43 | 35.5 | 9.2 | **≈ 31.5 years** |

Following the left spine of the fitted tree gives the node default rates directly:

| Condition | Clients | Default rate |
|---|---|---|
| `PAY_0 <= 1.5` | 21,464 | 16.6% |
| … and `PAY_2 <= 1.5` | 19,642 | 14.2% |
| … and `PAY_AMT3 > 466` | 14,813 | **11.6%** |
| … and `PAY_AMT3 <= 466` | 4,829 | **22.2%** |
| `PAY_0 > 1.5` | 2,508 | **69.7%** |

So the learned rule, in business language, is roughly:

> **If the client is 2+ months delinquent, risk jumps to about 70% and stays there unless
> their third-month payment status was clean too. Among clients who are *not* currently
> delinquent, the strongest refinement is how much they actually paid three months ago:
> paying almost nothing (under about NT$466) roughly doubles the default rate from 11.6%
> to 22.2%, and a low credit limit and a low last statement push it higher still.**

That last part is genuinely non-obvious and is the kind of interaction a linear model
cannot express: the same payment amount means different things depending on delinquency
history.

---

## 5.3 Confusion matrices, side by side

| Model | TN | FP | FN | TP | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 4,544 | **123** | 996 | 330 | **0.7285** | 0.2489 | 0.9736 |
| Decision Tree | 4,421 | 246 | 836 | 490 | 0.6658 | 0.3695 | 0.9473 |
| KNN | 4,388 | 279 | 880 | 446 | 0.6152 | 0.3363 | 0.9402 |
| Naive Bayes | 3,255 | **1,412** | **437** | **889** | 0.3864 | **0.6704** | 0.6975 |
| SVM | 4,413 | 254 | 865 | 461 | 0.6448 | 0.3477 | 0.9456 |

Read the FP and FN columns as two different business costs:

- **FP = 123** (Logistic Regression) means 123 good customers wrongly flagged. Excellent
  for customer relations, terrible for catching fraud.
- **FN = 996** (Logistic Regression) means 996 defaulters sailed through unflagged. Each
  one is a written-off balance.

There is no configuration that minimises both. **This table *is* the precision–recall
trade-off**, made concrete in NT dollars.

---

## 5.4 Model by model

### Logistic Regression — confident and cautious

**Predicted behaviour:** smooth linear boundary, effectively no interactions, optimises
log loss under imbalance → conservative.

**Observed:** fewest false positives (123), lowest recall (0.2489), **highest precision
(0.7285)**. AUC 0.7160.

**Why the AUC is close but not top:** the true boundary really is largely monotonic in
`PAY_0` — late payers default more — and that is exactly what a linear model captures. It
fails only where the relationship is conditional (a high bill matters more when
delinquency is also rising). Those interactions are a minority of cases, so LR loses
only ~0.031 AUC to the tree.

**The calibration tell:** its precision is high and its recall low, but its AUC (0.7160)
is noticeably better than its F1 (0.3710) would suggest. That gap means the *ranking* is
decent while the *threshold* at 0.5 is badly chosen for this class balance — the model has
the information but the default cut-off discards it. §2.8 of `02_metrics.md` shows the
same effect for the tree.

**Implementation note:** the fit is Newton's method on the L2-penalised log loss, and it
converges in **7 iterations** with an intercept of −1.4687 — a shade above the log-odds of
the base rate (`log(0.2213/0.7787) = −1.259`), which is the regularisation pulling the
model toward the majority class.

---

### Decision Tree — the winner, for now

**Predicted behaviour:** captures `PAY_0` first (highest impurity gain), adds conditional
refinements, shallow depth stops it memorising.

**Observed:** best on all three headlining metrics — accuracy 0.8195, AUC 0.7472, MCC
0.4007 — with **44 leaves at depth 6**.

**Why `PAY_0` dominates feature importance (66.7%):** impurity gain is computed at the
root, where the population is the full 50/50-ish mix; a split that separates 69.5%
defaulters from 16.6% defaulters produces an enormous Gini drop. Once that split exists,
every later split operates inside much purer subpopulations and therefore has less room to
improve — mechanically, downstream features *cannot* accumulate comparable importance.
**This is a known bias of impurity-based importance:** it favours features that split
early, and it is unreliable when features are correlated (the tree may split on `PAY_0`
and never give `PAY_2` its due, even though `PAY_2` carries almost the same information).

**Why depth 6 rather than 4:** tuning on the validation split scored depth 4 at 0.4605 and
depth 6 at 0.4609 — a tie in practice — while depth 8 and beyond fell to 0.4432 and 0.4323.
The extra two levels buy a little refinement around the delinquency split without enough
capacity to memorise noise, and they show up on the test set as the best AUC of the five.
That the scores for depths 4 and 6 are indistinguishable is itself the honest finding: this
dataset has roughly five levels of usable signal and then it stops.

---

### KNN — decent ranking, expensive to serve

**Predicted behaviour:** local boundaries, needs `k` tuned, harmed by the curse of
dimensionality.

**Observed:** AUC 0.7333 (just behind the tree's 0.7472), MCC 0.3520.

**Why its recall is middling (0.3363):** with distance weighting and `k=15`, a query near
the boundary is outvoted by the ~11 non-defaulters typically among its 15 neighbours. This
is the same minority-dilution effect that made `k=75` perform worst in tuning (validation
F1 0.3940 versus 0.4365 at `k=15`).

**Why its AUC is nevertheless near the top:** KNN is a pure ranking method — it never
assumes a functional form. If two clients have similar `PAY_0`, similar bills and similar
payments, KNN gets it right regardless of how those variables interact. That flexibility
shows up in AUC while the fixed 0.5 threshold still under-fires.

**The practical objection:** the artifact is 4.6 MB and every prediction is a 23,972-point
distance computation. The tree answers the same question from a 5 KB file in constant
time. Equal accuracy with ~1,000× the serving cost is not a tie in practice.

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

**Observed:** accuracy 0.8133 (level with Logistic Regression, 0.6 points behind the
tree), MCC 0.3755 (second best), AUC 0.7105 (lowest of the five).

**Why 1,877 of 4,000 points became support vectors (47%):** in a well-separated problem
only a handful of points sit on the margin. Here almost half do, which is a direct
measurement of how much the classes overlap — most clients are genuinely ambiguous. That's
the same fact the `PAY_0` table in §5.1 shows from a different angle. (The 4,000 is a
stratified subsample of the 23,972 training rows: the precomputed kernel matrix is
quadratic in memory, and the full set would need ~4.6 GB.)

**Why its AUC (0.7105) trails the tree's (0.7472) despite similar accuracy:** `C=1.0` and
`gamma='scale'` are the defaults, never tuned, and the model also sees only one sixth of
the training rows. A coarse grid over `C ∈ {0.1, 1, 10}` and `gamma ∈ {0.01, 0.1, 1}`
would almost certainly close that gap. We left it untuned to keep the five-model
comparison to a common, defensible procedure — but the headroom is real and worth saying
out loud.

**Why probabilities here are not calibratable by construction:** `predict_proba` returns
`sigmoid(decision_value)`. That is a monotone transform, so the *ranking* — and therefore
AUC, precision at a given threshold, and the confusion matrix — is exact, but the numbers
are not true probabilities. Logistic Regression and Naive Bayes produce coherent
probabilities; the SVM here does not, and should not be used to price risk.

---

## 5.5 Why four of the five cluster around 0.81 accuracy

| Model | Accuracy | vs. 0.7787 baseline |
|---|---|---|
| Decision Tree | 0.8195 | +0.0408 |
| SVM | 0.8133 | +0.0346 |
| Logistic Regression | 0.8133 | +0.0346 |
| KNN | 0.8066 | +0.0279 |
| Naive Bayes | 0.6915 | **−0.0872** |

Four models land within **1.3 percentage points of each other** — and SVM and Logistic
Regression land on *identical* accuracy. That is not a coincidence and it is not a bug —
it is the expected outcome when:

1. The positive class is 22% of the data, so ~78% of every prediction is "no default" and
   is right for free.
2. The signal is concentrated in one column (`PAY_0`). Every algorithm finds the same
   dominant feature; they differ only in how they refine around it.

**This is why the assignment asks for six metrics.** Accuracy cannot separate SVM from
Logistic Regression at all (both 0.8133), and it ranks them above KNN by 0.7 points. On
MCC the same three models are 0.3755 / 0.3494 / 0.3520 — a 2.6-point spread between the
best and worst, which is a far more meaningful separation than accuracy provides.

---

## 5.6 But is the Decision Tree *really* the winner?

Bootstrap resampling — 300 resamples of the 5,993 test rows with replacement — gives
sampling intervals:

| Model | AUC [95% CI] | MCC [95% CI] |
|---|---|---|
| Logistic Regression | 0.7160 [0.6998, 0.7331] | 0.3494 [0.3185, 0.3807] |
| **Decision Tree** | **0.7472 [0.7329, 0.7629]** | **0.4007 [0.3643, 0.4283]** |
| KNN | 0.7333 [0.7186, 0.7505] | 0.3520 [0.3234, 0.3869] |
| Naive Bayes | 0.7283 [0.7125, 0.7471] | 0.3140 [0.2883, 0.3381] |
| SVM | 0.7105 [0.6946, 0.7270] | 0.3755 [0.3463, 0.4070] |

Read this honestly:

- **The Decision Tree's AUC interval clears the SVM's** (0.733–0.763 vs 0.695–0.727). That
  one comparison *is* statistically defensible: on ranking quality, the tree genuinely
  beats this SVM. The tree against Logistic Regression is borderline — the intervals
  overlap by 0.0002 — so do not claim it.
- **Tree vs KNN and tree vs Naive Bayes overlap on AUC.** Those rankings are ties. The
  apparent ordering 0.7472 > 0.7333 is inside the noise.
- **On MCC, the Decision Tree and Naive Bayes intervals do not overlap** (0.364–0.428 vs
  0.288–0.338). That comparison is defensible too.
- **Tree vs SVM and tree vs KNN overlap** on MCC. Those are ties.

So the accurate statement is:

> The Decision Tree has the best point estimate on every headline metric, and its advantage
> in ranking quality over the SVM and in skill over Naive Bayes both survive bootstrap
> resampling. Against KNN, Logistic Regression and the SVM on MCC, it is statistically
> indistinguishable. Its decisive practical advantages are interpretability and a 5 KB
> artifact rather than a measurable accuracy lead.

That is a much stronger claim than "the tree wins", because it survives scrutiny. Saying
precisely what your evidence supports is worth more in a viva than an overconfident
ranking.

---

## 5.7 What would actually improve these numbers, in order

1. **Tune the operating threshold** (§2.8). Free. Dropping the tree's cut to 0.25 lifts F1
   from 0.4753 to 0.5136 and recall from 0.3695 to 0.5905 — but costs precision
   (0.6658 → 0.4544) and MCC (0.4007 → 0.3568). Whether that trade is worth it is a
   business decision, not a statistical one. This is the highest value-per-line-of-code
   change available.
2. **Add class weighting to the loss** in LR, SVM and the tree. Reweights the objective so
   the minority class is no longer systematically under-predicted; typically doubles recall
   at a precision cost. In this codebase that means a per-sample weight in the gradient
   (`models/logistic.py`), the Gini computation (`models/tree.py`) and the SMO
   sub-problem (`models/svm.py`).
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
- **The SVM trains on a 4,000-row stratified subsample**, not the full 23,972 rows, because
  the precomputed kernel matrix is quadratic in memory. `validate_models.py` shows the SMO
  implementation matches scikit-learn exactly on that same subsample, so this is a limit on
  the *model*, not a bug in the *implementation* — but it is a real handicap in the
  comparison and should be stated rather than buried.
- **The tree's candidate thresholds are quantiles, not exhaustive midpoints.** This is the
  one deliberate algorithmic approximation; it costs 0.0035 AUC against scikit-learn's
  exact search, which `validate_models.py` reports directly.
- **Single 80/20 split.** Cross-validation would give tighter estimates, though the
  bootstrap in §5.6 partly compensates.
- **No `random_state` dependence check.** All results use seed 42; a different seed would
  move the numbers a little. The bootstrap intervals indicate how much.
- **`feature_importances_` is impurity-based** and therefore biased toward
  high-cardinality, early-splitting features and unreliable under correlation. Permutation
  importance would be the more robust measure.
