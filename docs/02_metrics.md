# 2. Metrics — What Each Number Means and How Ours Was Computed

Every metric in this project is computed from one 2×2 table. Get the table right and
everything else is arithmetic.

## 2.1 The confusion matrix

For our Decision Tree on the 5,993 held-out clients:

|  | Predicted **no default** | Predicted **default** | Row total |
|---|---|---|---|
| **Actually no default** | TN = 4,402 | FP = 265 | 4,667 |
| **Actually default** | FN = 836 | TP = 490 | 1,326 |
| **Column total** | 5,238 | 755 | 5,993 |

- **TP (true positive)** — flagged as a defaulter, and did default. The win.
- **FP (false positive)** — flagged, but paid fine. A wrongly penalised customer.
- **FN (false negative)** — not flagged, but defaulted. The bank eats the loss.
- **TN (true negative)** — not flagged, paid fine.

Sanity check the totals: $4{,}402 + 265 = 4{,}667$ actual non-defaulters, and
$4{,}667/5{,}993 = 0.7787$ — matching the known 77.9% base rate. Good.

---

## 2.2 Accuracy

$$\text{Accuracy} = \frac{TP + TN}{N} = \frac{490 + 4{,}402}{5{,}993} = \frac{4{,}892}{5{,}993} = \mathbf{0.8163}$$

### Why this number is nearly meaningless here

Predict "no default" for all 5,993 clients. Then $TP = 0$, $TN = 4{,}667$:

$$\text{Accuracy} = \frac{0 + 4{,}667}{5{,}993} = 0.7787$$

**A model that has learned nothing scores 0.7787.** Our best model scores 0.8163.

So the honest way to read our results is not "81.6% accurate" but:

> "3.8 percentage points better than doing nothing."

Any viva answer that quotes 0.8163 without this comparison is a red flag to an examiner.

---

## 2.3 Precision

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{490}{490 + 265} = \frac{490}{755} = \mathbf{0.6490}$$

> **Of the 755 clients we flagged, 65% actually defaulted.**

The question precision answers: *when the model raises an alarm, how often is it right?*
This is the metric that governs how much you annoy good customers.

---

## 2.4 Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{490}{490 + 836} = \frac{490}{1{,}326} = \mathbf{0.3695}$$

> **Of the 1,326 clients who actually defaulted, we caught 37%. We missed 836.**

This is the metric a credit-risk team actually cares about, because FN is the costly
error. Our tree misses 63% of defaulters — which is the uncomfortable headline of this
project, and worth stating plainly rather than hiding.

### Specificity — the flip side

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{4{,}402}{4{,}667} = 0.9432$$

94% of good customers are correctly left alone.

---

## 2.5 F1 Score

$$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
= \frac{2(0.6490)(0.3695)}{0.6490 + 0.3695} = \frac{0.4796}{1.0185} = \mathbf{0.4709}$$

F1 is the **harmonic mean**, not the arithmetic mean. The harmonic mean punishes
imbalance: with $P = 1.0, R = 0.1$ the arithmetic mean is 0.55 but F1 is only 0.18. So
F1 only gets high when *both* precision and recall are high — which is exactly the
property that makes it useful for rare-event problems.

---

## 2.6 AUC — Area Under the ROC Curve

### Building the curve

A classifier that outputs a probability can be made more or less aggressive by moving
the decision threshold. Sweep it from 1.0 down to 0.0 and for each value compute:

$$\text{TPR} = \frac{TP}{TP+FN} \ (\text{recall}), \qquad
\text{FPR} = \frac{FP}{FP+TN}$$

Plot TPR against FPR; **AUC is the area under that curve.**

| Threshold | TPR (recall) | FPR | Behaviour |
|---|---|---|---|
| 1.0 | 0.00 | 0.00 | Flag nobody |
| 0.5 | 0.37 | 0.057 | Our operating point |
| 0.0 | 1.00 | 1.00 | Flag everybody |

A random classifier traces the diagonal, area **0.5**. A perfect one hits the top-left
corner, area **1.0**. Ours is **0.7341**.

### The interpretation worth memorising

AUC equals the probability that a randomly chosen defaulter is ranked riskier than a
randomly chosen non-defaulter. **0.7341 → the model puts the right client first 73% of
the time.**

### Why AUC is preferred over accuracy here

- It is **threshold-free** — it judges the quality of the ranking, not one arbitrary cut.
- It is **insensitive to the 78/22 imbalance** in a specific sense: it compares positives
  against negatives rather than measuring against the majority class. Accuracy is
  dominated by the majority; AUC is not.
- It is **not** a complete story. A model can rank well and still pick a bad threshold —
  which is exactly what happened to us (see §2.8).

---

## 2.7 MCC — Matthews Correlation Coefficient

### The formula

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Plugging in our tree's numbers:

- Numerator: $490 \cdot 4{,}402 - 265 \cdot 836 = 2{,}156{,}980 - 221{,}540 = 1{,}935{,}440$
- Denominator: $\sqrt{755 \cdot 1{,}326 \cdot 4{,}667 \cdot 5{,}238}
  = \sqrt{2.4473\times10^{13}} = 4{,}947{,}057$
- $\text{MCC} = 1{,}935{,}440 / 4{,}947{,}057 = \mathbf{0.3912}$ ✓

### What it means

MCC is the **correlation between the prediction and the truth**, treating both as binary
variables. Range: $[-1, +1]$.

| MCC | Meaning |
|---|---|
| +1.0 | Perfect prediction |
| 0.0 | Completely uncorrelated with the truth — no better than chance |
| −1.0 | Perfectly wrong (systematically inverted) |

Note the crucial property: the numerator is $TP \cdot TN - FP \cdot FN$. It only gets
large when **both** $TP$ and $TN$ are large relative to the errors. A model that predicts
one class almost always — like Naive Bayes — cannot get a high MCC, no matter how good
its recall looks.

### Our range of MCC values

| Model | MCC | Interpretation |
|---|---|---|
| Decision Tree | 0.3912 | Best available on this data — moderate |
| SVM | 0.3794 | Close second |
| KNN | 0.3545 | Moderate |
| Logistic Regression | 0.3496 | Moderate |
| Naive Bayes | 0.3140 | Weakest, despite highest recall |

**A "good" MCC depends entirely on the domain.** 0.39 for credit default is respectable
— published models on this exact dataset typically report AUC 0.70–0.78. What would be
alarming is an MCC of 0.9 on a task where the best-known results are 0.4, because it
would imply you had leaked the answer.

### Why MCC over F1?

Both use all four cells, but F1 ignores $TN$ entirely. In a 78/22 problem, $TN$ is almost
the entire dataset, so ignoring it hides almost everything. MCC uses all four. That is
why it is the metric most resistant to being gamed by imbalance.

### Worked contrast: Naive Bayes

| | Decision Tree | Naive Bayes |
|---|---|---|
| TN | 4,402 | 3,255 |
| FP | 265 | **1,412** |
| FN | 836 | 437 |
| TP | 490 | **889** |
| Precision | 0.6490 | 0.3864 |
| Recall | 0.3695 | **0.6704** |
| F1 | 0.4709 | **0.4902** |
| MCC | **0.3912** | 0.3140 |

Naive Bayes wins on F1 and recall, yet loses on MCC. It achieved high recall by wildly
over-predicting default — 1,412 false alarms. **F1 alone would have picked the wrong
model.** This table is the best single argument for including MCC in any report.

---

## 2.8 The threshold problem — a finding you should know about

scikit-learn's `predict()` uses a fixed 0.5 cut. That is a **default, not a law.** Here
is our Decision Tree swept across thresholds on the test set:

| Threshold | Flagged | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| 0.20 | 1,778 | 0.4438 | 0.5950 | 0.5084 | 0.3481 |
| 0.25 | 1,778 | 0.4438 | 0.5950 | 0.5084 | 0.3481 |
| 0.30 | 1,070 | 0.5673 | 0.4578 | 0.5067 | 0.3886 |
| **0.40** | 863 | 0.6234 | 0.4057 | **0.4915** | **0.3974** |
| **0.50** ← used | 755 | **0.6490** | 0.3695 | 0.4709 | 0.3912 |
| 0.60 | 496 | 0.7077 | 0.2647 | 0.3853 | 0.3520 |

Two things to notice:

1. **Rows 0.20 and 0.25 are identical.** A tree's `predict_proba` is the fraction of
   training positives that landed in a leaf — only 15 leaves means only a handful of
   distinct probability values exist. Thresholds between two adjacent leaf values do
   nothing. This is a real, observable difference between trees and smooth models.
2. **0.40 beats 0.50 on both F1 (0.4915 vs 0.4709) and MCC (0.3974 vs 0.3912).**
   Meanwhile accuracy *prefers* 0.50 (0.8163 vs 0.8143). Which threshold is "correct"
   depends entirely on which metric you were told to optimise — and on the relative cost
   of an FN (a written-off loan) versus an FP (an annoyed customer).

We report 0.5 because the assignment asks for standardised comparison across five models.
But if you are asked "how would you improve this?", this table *is* the answer: choose
the threshold from a cost matrix, not by accepting the default.

---

## 2.9 Metric summary table

| Metric | Our tree | Range | Answers | Blind spot |
|---|---|---|---|---|
| Accuracy | 0.8163 | 0–1 | Overall hit rate | Majority class dominates; floor is 0.7787 |
| AUC | 0.7341 | 0–1 | Ranking quality | Says nothing about the chosen threshold |
| Precision | 0.6490 | 0–1 | How often alarms are right | Ignores missed defaulters |
| Recall | 0.3695 | 0–1 | How many defaulters caught | Ignores false alarms |
| F1 | 0.4709 | 0–1 | Balance of P and R | Ignores TN entirely |
| MCC | 0.3912 | −1–+1 | Correlation with truth | Harder to explain to stakeholders |

Next: [03_data_pipeline.md](03_data_pipeline.md) — how the raw spreadsheet became the
matrix the models saw.
