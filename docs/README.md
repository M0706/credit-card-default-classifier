# Study Notes — Credit Card Default Prediction

These documents explain the project from first principles. They assume you know
Python and basic statistics, but no machine learning. Every number quoted here was
produced by the code in this repository against the committed `test_data.csv`, so you
can re-derive any of them yourself.

## Reading order

| # | Document | What it answers |
|---|---|---|
| 1 | [01_theory.md](01_theory.md) | What *is* a classifier, and how do the five algorithms actually work? |
| 2 | [02_metrics.md](02_metrics.md) | What do accuracy, AUC, precision, recall, F1 and MCC mean — and why is accuracy nearly useless here? |
| 3 | [03_data_pipeline.md](03_data_pipeline.md) | Raw `.xls` → cleaned CSV → scaled training matrix. Every decision and its justification. |
| 4 | [04_code_walkthrough.md](04_code_walkthrough.md) | Line-by-line tour of every script, including the `models/` package. |
| 5 | [05_why_these_results.md](05_why_these_results.md) | Why each model produced *its* specific numbers, read off the trained artefacts. |
| 6 | [06_viva_prep.md](06_viva_prep.md) | 41 questions you should be able to answer about this submission. |
| 7 | [07_implementations.md](07_implementations.md) | The five algorithms written from scratch: the maths in each file, the bugs it is easy to make, and the proof they match scikit-learn. |

## The 60-second version

1. We have 29,965 bank clients described by 23 numbers, and we know whether each
   defaulted on their next payment (22.1% did).
2. We hide 20% of them (5,993 clients) and never let any model see those labels
   during training. That held-out set is `test_data.csv`.
3. Five different algorithms learn the mapping `23 numbers → default / no default`
   using only the other 23,972 clients.
4. We score each on the hidden 20% using six metrics, because one metric can always
   be gamed.
5. A Streamlit app loads the five saved models and re-scores any CSV you upload —
   without retraining, so its numbers are honest held-out estimates.

## The one number to remember

Predicting "no default" for **every single client** scores **0.7787 accuracy**.
Four of the five models reached only ~0.81. Any accuracy figure in this project must be
read against that 0.7787 floor, not against zero. This single fact is why the
assignment asks for MCC.

## Reproducing everything

```bash
python prepare_data.py     # .xls -> data/credit_default.csv (29,965 rows)
python train_models.py     # trains 5 models -> model/, reports/, test_data.csv
python validate_models.py  # from-scratch vs scikit-learn -> reports/validation.json
python analysis.py         # bootstrap intervals + threshold sweep
streamlit run app.py       # interactive evaluation
```

Both training runs are deterministic: `random_state=42` fixes the train/test split,
the validation split, the decision-tree tie-breaking, and SVM's internal shuffling.
Re-running produces byte-identical metrics.
