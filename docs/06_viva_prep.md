# 6. Viva / Defence Preparation

Thirty questions an examiner could reasonably ask, with answers grounded in the actual
numbers from this project.

---

## Dataset and preprocessing

**1. Why this dataset, and does it meet the assignment requirements?**
UCI Default of Credit Card Clients: 29,965 rows after cleaning, **23 features**, binary
target. Requirements were ≥12 features and ≥500 instances — met with margin. It's a
genuine credit-risk problem with 22.1% positive rate, so it exercises every metric
properly rather than producing five identical 0.96 scores.

**2. Why did you remove 35 rows?**
They were exact duplicates. Duplicates get double weight in training, and if one copy
lands in train and another in test, the model has effectively seen a test client — a
subtle leak. `drop_duplicates()` runs **before** the split so a duplicate can never
straddle both partitions. 30,000 → 29,965.

**3. Why fold EDUCATION codes 0, 5 and 6 into "others"?**
The UCI documentation only defines 1–4. Codes 0 (14 rows), 5 (280) and 6 (51) are
undocumented — 345 rows carrying values the dataset's authors never defined. Three
categories with 14/280/51 examples are too small to learn from and can't be justified in a
report. Same reasoning for `MARRIAGE = 0` (54 rows). The alternative was deleting those
rows, which throws away valid credit and repayment information.

**4. `BILL_AMT` has negative values. Is that corrupt data?**
No. A negative bill is a **credit balance** — the client overpaid or received a refund.
`BILL_AMT6` reaches −339,603. Deleting these would systematically remove the most
creditworthy customers.

**5. What do `PAY_0` … `PAY_6` mean?**
Repayment status for the most recent month back to six months ago: −2 = no consumption,
−1 = paid in full, 0 = revolving credit, 1–8 = months of delay. **It's the strongest
predictor in the dataset** — `PAY_0 >= 2` covers 10.4% of clients and they default at
69.5%, versus 16.6% for everyone else.

**6. Why is the header read with `header=1`?**
Row 0 of the spreadsheet is a dummy header (`X1, X2, … default payment next month`); the
real column names are on row 1. Reading with the default `header=0` produces `X1…X23`
column names and one junk record.

---

## Splitting and leakage

**7. Why 80/20 rather than 70/30 or cross-validation?**
80/20 gives 5,993 test rows — enough for stable metrics — while leaving 23,972 for
training. Cross-validation would give tighter estimates, but a single held-out set is
simpler to explain and matches the assignment's requirement for a committed `test_data.csv`.

**8. What does `stratify=y` do and why does it matter?**
It forces every partition to have the same 22.1% positive rate. Without it, a 20% sample
could land at 20.5% or 21.7% positive by chance, shifting every metric for reasons unrelated
to model quality.

**9. What is data leakage and where could it have occurred here?**
Leakage is when information from the test set influences training, making the score
optimistic. Three places it could have happened:
- Fitting the `StandardScaler` on all rows before splitting → we fit on the fit-split only
  (19,177 rows), then `transform` everywhere.
- Duplicates spanning train and test → removed before splitting.
- Tuning on the test set → we tune on a validation slice carved out of training only.

**10. Why does `train_models.py` make two splits?**
The inner split (19,177 fit / 4,795 validation) exists to choose the tree depth and KNN
`k`. Comparing candidates on the test set would be tuning on the answer key. After
selection, the winner is **refit on all 23,972 training rows** so the validation rows
aren't wasted.

**11. Why must the scaler be saved rather than recomputed in the app?**
`StandardScaler` stores a mean and standard deviation per feature. At inference, uploaded
rows must be divided by **those** numbers. Recomputing from the upload would rescale the
same client differently depending on who else is in the file — every prediction would be
meaningless. It's the most common deployment bug in projects like this.

**12. Why keep `metadata.json`?**
It records the 23 column names **in training order**. `scaler.transform` applies column 0's
mean to column 0 of whatever array it receives, so a CSV with identical columns in a
different order would produce silently wrong predictions — no error, just garbage. The app
does `frame[FEATURES]` to enforce the order.

---

## Scaling

**13. Why is scaling necessary?**
`LIMIT_BAL` spans 10,000–1,000,000 while `PAY_0` spans −2–8. In a squared Euclidean
distance, `LIMIT_BAL` contributes up to ~10¹² and `PAY_0` contributes ~1. Unscaled, the
single most predictive column becomes invisible to KNN and SVM.

**14. Which models need it and which don't?**
Need: Logistic Regression (optimiser stability), KNN (distance), SVM (kernel is a function
of distance). Don't need: Decision Tree (only compares against thresholds; any monotonic
rescaling preserves ordering), Gaussian NB (models each feature independently). We scale for
all five anyway so every model sees identical inputs — that's what makes the comparison
fair.

**15. What exactly does standardisation do?**
$z = (x - \mu)/\sigma$ per feature, using $\mu$ and $\sigma$ from the **fit split**. After
it, every feature has mean 0 and SD 1, so $z = 2$ means "two SDs above average" in any
column.

---

## The algorithms

**16. Explain Logistic Regression in one sentence.**
It models the log-odds of default as a linear function of the features:
$\log\frac{p}{1-p} = w \cdot x + b$, so the decision boundary is a hyperplane and each
coefficient is an odds multiplier.

**17. Interpret your largest coefficient.**
`PAY_0` = +0.6535. Since features are standardised, one SD more delinquency in the most
recent month multiplies the odds of default by $e^{0.6535} = 1.92$, holding all else fixed.
The intercept −1.4676 gives a base probability of 0.187, close to the 22.1% base rate.

**18. How does a Decision Tree choose splits?**
Greedily: at each node, try every feature and threshold and pick the split that most
reduces **Gini impurity**, $1 - \sum p_k^2$, weighted by child size. Then recurse. It is
greedy and has no lookahead, so the tree is not globally optimal.

**19. How does KNN predict?**
`fit` just stores the data. Prediction finds the 15 nearest training points by Euclidean
distance (on standardised features) and lets them vote, weighted by $1/\text{distance}$.

**20. What is the curse of dimensionality?**
In 23 dimensions distances concentrate — the ratio between nearest and farthest neighbour
approaches 1, so "nearest" stops meaning "similar". It's why KNN is mid-table here despite
being assumption-free.

**21. What makes Gaussian Naive Bayes "naive"?**
It assumes features are conditionally independent given the class:
$P(x|y) = \prod_i P(x_i|y)$. That's false here — `BILL_AMT1…6` are six measurements of
nearly the same thing.

**22. What does that violation do to your results, concretely?**
It multiplies the same evidence six times, inflating the posterior for class 1. Consequence:
highest recall (0.6704) but **1,412 false positives** versus 265 for the tree, precision
0.3864, and accuracy 0.6915 — **worse than the 0.7787 "predict nobody defaults" baseline**.

**23. What is the kernel trick?**
Instead of explicitly mapping features into a high-dimensional space, replace the dot
product with a kernel function $K(x,x')$. The RBF kernel
$K = \exp(-\gamma\lVert x-x'\rVert^2)$ implicitly maps to infinite dimensions, letting a
linear separator exist there without ever computing the mapping.

**24. What are support vectors, and how many did you get?**
The training points with $\alpha_i > 0$ — those on or violating the margin. Only they
contribute to predictions. We got **10,506 of 23,972 (44%)**, which is high, and is a direct
measure of how much the classes overlap.

**25. What does `C` control?**
The penalty for margin violations. Small `C` = wide margin, more violations tolerated (more
regularisation). Large `C` = narrow margin, tighter fit (overfitting risk). We used the
default 1.0.

**26. Why did you use `decision_function` for SVM's AUC?**
`SVC(probability=False)` has no `predict_proba`. AUC only needs a risk *ranking*, and the
signed distance to the boundary is a valid ranking, so `decision_function` gives identical
AUC without the slow internal Platt-scaling step.

---

## Metrics

**27. Why is accuracy a bad metric here?**
Predicting "no default" for everyone scores **0.7787**. Four of five models reach only
~0.81. Accuracy is dominated by the 78% majority class, so it hides the fact that we miss
63% of defaulters.

**28. What does MCC add over F1?**
MCC uses all four confusion cells:
$(TP\cdot TN - FP\cdot FN)/\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}$. F1 ignores $TN$ entirely —
and in a 78/22 problem $TN$ is almost the whole dataset. Naive Bayes beats the tree on F1
(0.4902 vs 0.4709) yet loses on MCC (0.3140 vs 0.3912) because of its 1,412 false
positives. **F1 alone would have picked the wrong model.**

**29. What does AUC 0.7341 mean in plain English?**
Pick a random defaulter and a random non-defaulter. The model ranks the defaulter as
riskier 73.4% of the time. A coin flip gives 50%.

**30. Your tree's MCC is 0.39. Is that good?**
Moderate, and appropriate. Published models on this dataset report AUC roughly 0.70–0.78,
so 0.73 is in line. An MCC of 0.9 here would be *suspicious* — it would suggest leakage.
MCC is domain-relative; 0.39 on a noisy behavioural dataset is a real signal.

---

## Results, honesty, and improvement

**31. Which model wins, and how confident are you?**
The Decision Tree has the best point estimate on accuracy (0.8163), AUC (0.7341) and MCC
(0.3912). But bootstrap 95% intervals overlap for all five on AUC, and overlap for tree vs
SVM/KNN on MCC. **The only statistically defensible claim is that the tree clearly beats
Naive Bayes on MCC** (0.360–0.418 vs 0.287–0.338). The tree is chosen for interpretability
and a 3 KB artifact, not a proven accuracy lead.

**32. Why do four models cluster within 0.9 points of accuracy?**
Because ~78% of predictions are "no default" and correct for free, and the signal is
concentrated in `PAY_0` — every algorithm finds the same dominant feature and differs only
in refinement. This is exactly why six metrics were requested.

**33. Your recall is only 0.37. Is the model useful?**
Not as-is for production. Missing 63% of defaulters is operationally unacceptable. But the
project *measures* that rather than hiding it, and §2.8 of `02_metrics.md` shows a free
improvement: at threshold 0.40, MCC rises to 0.3974 and F1 to 0.4915 with no retraining.

**34. How would you improve it?**
In value order: (1) tune the operating threshold from a cost matrix; (2)
`class_weight="balanced"`; (3) tune SVM `C`/`gamma` (currently defaults); (4) engineer
features — utilisation ratio `BILL_AMT1/LIMIT_BAL`, average lateness, repayment fraction;
(5) one-hot encode `PAY_*` as ordered categories; (6) gradient boosting, which typically
reaches AUC ≈ 0.78 on this dataset.

**35. What is overfitting, and where did you observe it?**
Letting a model learn noise that won't recur. Observable in the tuning output: validation
F1 **fell** as tree depth increased (0.4628 at depth 4 → 0.4290 unlimited), even though
training accuracy rose to 1.000. More capacity made it worse — that measured decline is the
evidence.

**36. Why does `drop_duplicates()` come before the split and not after?**
If it runs after splitting, duplicates could already be on both sides. Removing them
beforehand guarantees a client appears in exactly one partition.

**37. Your app must not retrain. How do you guarantee that?**
`app.py` never imports an estimator class or `train_test_split`, and calls only `predict` /
`predict_proba` / `decision_function`. It loads `model/*.joblib` and
`model/scaler.joblib` and applies the saved transform. If `.fit(` appeared in the file, the
requirement would be violated.

**38. What happens if the uploaded CSV has the columns in a different order?**
Nothing bad — the app selects `frame[FEATURES]` by name using the order stored in
`metadata.json`. If a required column is *missing*, it shows an error and calls `st.stop()`
rather than letting a `KeyError` surface.

**39. Why is `scikit-learn` pinned to an exact version in `requirements.txt`?**
The models are pickled `.joblib` files. Unpickling into a different scikit-learn version
can raise `InconsistentVersionWarning` or fail outright, because internal attribute layouts
change between releases. Deployment reproducibility requires the same version.

**40. What was the hardest bug in this project?**
The silently-under-trained models. The tuning helper fits candidates on the 80% fit-split;
the first version then skipped refitting the two tuned models, so the Decision Tree and KNN
were scored while trained on only 19,177 of 23,972 rows — 20% of the data thrown away. It
was invisible in the printed metrics; it surfaced only by inspecting artefacts
(`knn._fit_X.shape == (19177, 23)`, tree with 13 leaves). Fixing it raised the tree's MCC
from 0.3858 to 0.3912 and *lowered* KNN's from 0.3611 to 0.3545, which is why you measure
rather than assume.
