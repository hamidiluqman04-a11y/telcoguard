# TelcoGuard — Telco customer churn prediction with marimo

TelcoGuard predicts which telecom customers are likely to cancel their
service ("churn"). It explains why, and suggests retention actions worth
testing. The whole project runs in [marimo](https://marimo.io) notebooks. Each
notebook is a plain Python file you can open as an interactive notebook, serve
as a read-only app, or run as a script.

Built with Python, marimo, pandas, NumPy, scikit-learn, Plotly, joblib and
pytest. There is no separate web framework.

---

## Results at a glance

All numbers come from the notebooks' own output on the real dataset (7,043
customers, 26.54% churn).

| | Logistic Regression | Random Forest | **Hist. Gradient Boosting** (final) |
|---|---|---|---|
| CV PR-AUC (5-fold, train) | 0.661 | 0.663 | **0.668** |
| Test PR-AUC (95% CI) | 0.634 (0.580–0.683) | 0.653 (0.598–0.703) | **0.663** (0.609–0.712) |
| Test ROC-AUC (95% CI) | 0.842 (0.819–0.863) | 0.844 (0.821–0.864) | **0.849** (0.828–0.869) |

Final model at its decision threshold of **0.30** (chosen on training data
only), on the 1,409 held-out test customers:

| Precision | Recall | F1 | Customers flagged | TP / FP / FN / TN |
|---|---|---|---|---|
| 53.6% | 77.5% | 0.634 | 38.4% | 290 / 251 / 84 / 784 |

In plain words: the model catches about **3 in 4 churners**. About **half of
the customers it flags** would actually have churned. The three models
perform almost identically, and their confidence intervals overlap.

> Re-running the notebooks reproduces these numbers exactly: same split, same
> models, same predictions. This was verified by rebuilding everything from
> scratch and comparing.

---

## Project structure

```
telcoguard/
├── data/
│   ├── Telco-Customer-Churn.csv        # the original dataset (input, never modified)
│   └── processed/train_test_split.csv  # which customers are train vs test (written by notebook 02)
├── notebooks/
│   ├── 01_data_exploration.py          # data quality + interactive EDA
│   ├── 02_data_preprocessing.py        # cleaning, stratified split, leakage-safe pipeline
│   ├── 03_model_training.py            # 3 models tuned with cross-validation, saved to models/
│   ├── 04_model_evaluation.py          # held-out test evaluation, thresholds, final choice
│   └── 05_churn_insights.py            # interactive app: scoring, explanations, suggestions
├── src/
│   ├── config.py                       # paths, random seed, expected schema
│   ├── data_processing.py              # load, validate, clean, split, preprocessor
│   ├── training.py                     # model grids, CV tuning, save/load artifacts
│   ├── evaluation.py                   # metrics, bootstrap CIs, threshold policies
│   ├── prediction.py                   # score customers, CSV upload, risk categories
│   └── explainability.py               # importance, per-customer explanations, actions
├── models/                             # fitted pipelines (*.joblib) + metadata (*.json)
├── reports/                            # test_threshold_results.csv
├── tests/                              # 118 pytest tests
├── requirements.txt
├── pytest.ini
└── README.md
```

**Differences from the originally planned layout, and why**

- `src/config.py` holds the seed, file paths and column lists in one place,
  so every notebook and test uses the same values.
- `src/evaluation.py` holds metrics and threshold logic. It is separate from
  `training.py` so the insights notebook can use it without the tuning code.
- `data/processed/train_test_split.csv` records the split as a file, so
  notebooks 03–05 provably use the *same* customers.
- The dataset file is `Telco-Customer-Churn.csv` (the copy supplied with the
  project). Its columns match the IBM sample file
  `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- `pytest.ini` lets tests import `src` and defines a `slow` marker.

---

## Setup

Requires **Python 3.13** (developed on 3.13.3). From the project folder:

**Windows (Git Bash or PowerShell)**

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` pins the exact versions the project was tested with.
Commands below use the Windows path `.venv/Scripts/...`; on macOS/Linux use
`.venv/bin/...` instead.

---

## Running the notebooks

The notebooks build on each other's outputs, so **run them in order the first
time**:

| # | Notebook | Reads | Writes | Run time* |
|---|---|---|---|---|
| 01 | data exploration | `data/*.csv` | nothing | ~6 s |
| 02 | preprocessing | `data/*.csv` | `data/processed/train_test_split.csv` | ~4 s |
| 03 | model training | split | `models/*.joblib`, `models/*.json` | ~1–2 min |
| 04 | model evaluation | split, models | `models/final_model.json`, `reports/*.csv` | ~20 s |
| 05 | churn insights | split, models, final model | nothing (downloads on request) | ~10 s |

\*On a 12-core laptop.

Model files (`*.joblib`, up to 8 MB) are **not committed to git**. After a
fresh clone, run notebooks 02–04 before opening notebook 05. Running a
notebook too early fails with a message telling you which notebook to run
first.

There are three ways to run a notebook:

```bash
# 1. Interactive editing (see and change the code)
.venv/Scripts/marimo.exe edit notebooks/01_data_exploration.py

# 2. As a read-only app (code hidden, widgets work); best for notebook 05
.venv/Scripts/marimo.exe run notebooks/05_churn_insights.py

# 3. As a plain script (runs every cell, fails on any error)
.venv/Scripts/python.exe notebooks/03_model_training.py
```

To rebuild everything from scratch:

```bash
for nb in 01_data_exploration 02_data_preprocessing 03_model_training 04_model_evaluation 05_churn_insights; do .venv/Scripts/python.exe notebooks/$nb.py || break; done
```

---

## Testing

```bash
# Fast unit tests (~5 s)
.venv/Scripts/python.exe -m pytest -m "not slow"

# Everything, including running all five notebooks end to end (~2 min)
.venv/Scripts/python.exe -m pytest

# marimo's own notebook checks (reactivity rules, formatting)
.venv/Scripts/marimo.exe check --strict notebooks/
```

The 118 tests cover:

- **Data:** the real file's known facts (shape, 11 blank `TotalCharges`, churn
  rate), strict number parsing, and schema validation that rejects each kind
  of bad value.
- **Preprocessing:** the split is stratified, reproducible and has no overlap;
  the scaler learns from training data only; blank charges are imputed as 0;
  unknown categories are handled.
- **Training:** seeds are fixed, `customerID` is never a model input,
  artifacts reload with identical predictions, and a scikit-learn version
  mismatch produces a warning.
- **Evaluation:** metrics match scikit-learn, cost calculations are checked by
  hand, and the final threshold is shown to come from training data, not the
  test set.
- **Prediction:** bad uploads (empty, wrong encoding, malformed, missing
  columns, invalid values, duplicate IDs) are rejected with clear errors.
- **Explainability:** explanations never create impossible customers, and
  suggestions follow the two-signal rule.
- **Notebooks:** all five run top to bottom, and their key results are checked.

The `slow` tests run the notebooks, which rewrites `data/processed/` and
`models/`. The content is identical each time, but the `created_utc` and
`fit_seconds` fields in `models/*.json` change, so git will show those files
as modified.

---

## How it works (key decisions)

### Data handling

- The dataset is loaded and **validated** against an expected schema: every
  column must be present, categories must be known values, and numbers must be
  non-negative.
- `TotalCharges` is stored as text because **11 cells contain a single
  space**. All 11 are customers with `tenure = 0` who have not been billed yet,
  so the blanks become `NaN` and are filled with **0** inside the model
  pipeline. Any other non-numeric value is an error, not a silent `NaN`.
- `customerID` is an identifier. It is kept only as the row index and **never
  used as a feature**.

### Preventing leakage

- **Stratified 80/20 split** (`random_state=42`), saved to a file and reused
  by every later notebook.
- Preprocessing (imputer, scaler, one-hot encoder) and the model are combined
  in **one scikit-learn `Pipeline`**. It is fitted on training data only, and
  refitted inside each cross-validation fold.
- The **entire fitted pipeline** is saved with joblib, so new data is
  processed exactly like the training data.

### Model selection

- Logistic Regression, Random Forest and `HistGradientBoostingClassifier` are
  tuned with **5-fold stratified cross-validation** on the training set only.
- Hyperparameters are chosen by **PR-AUC (average precision)**, which suits
  the imbalanced target (26.5% churn). Accuracy is never used to choose
  anything: a model that predicts "nobody churns" is 73.5% accurate.
- The final model is the one with the best **cross-validated** PR-AUC. This
  was decided *before* looking at test results.

### Threshold

- At the default 0.5 threshold, the model misses about half of churners.
- The threshold of 0.30 maximises F1 on **out-of-fold training
  predictions**. The test set is used only to *report* results.
- Notebook 4 includes a cost calculator for your own campaign economics. Its
  default numbers are placeholders, not data.

### Explanations and suggestions

- **Overall importance:** how much test PR-AUC drops when each feature is
  shuffled (permutation importance). Tenure and contract type dominate.
- **Per-customer explanation:** how the predicted risk would change if one
  feature had the most common value. Tenure and TotalCharges change together,
  and impossible combinations are skipped.
- **Retention suggestions:** an action is shown only if the model's estimated
  risk drops by at least a chosen minimum **and** customers with the proposed
  value churned less historically.

---

## Using the insights app (notebook 05)

```bash
.venv/Scripts/marimo.exe run notebooks/05_churn_insights.py
```

1. **Choose customers.** Test customers are the default and give honest
   scores. "All customers" includes training customers, whose scores are
   optimistic. "Uploaded CSV" scores your own file.
2. **Set risk cut-offs.** The default is Low < 30% ≤ Medium < 60% ≤ High;
   Medium starts at the model threshold.
3. **Filter and sort** the customer table, then **click a row** to explain
   that customer and see suggested actions.
4. **Download** the filtered or full predictions as CSV.

**CSV upload format.** The file needs the same columns as the original
dataset. `customerID` and `Churn` are optional, and extra columns are ignored.
The app offers an example CSV and lists every allowed value. A file with any
problem is rejected as a whole, with a list of what to fix.

---

## Limitations

- **Estimated risk is not certainty.** A 70% risk means about 7 in 10
  *similar* customers churn. Any individual may not.
- **Association is not causation.** The model, the explanations and the
  suggestions all describe patterns in historical data. For example, customers
  who *choose* two-year contracts differ from those who don't. Test retention
  actions with a controlled experiment before relying on them.
- **The models are statistically tied.** The "final model" choice matters
  little; Logistic Regression is a reasonable, simpler alternative.
- **Snapshot data.** There are no dates, so the model cannot track changes over
  time. Monitor performance on new data and retrain when it drops.
- **What-if scenarios hold everything else fixed**, including price. Real
  offers usually change the price as well.
- **No fairness review yet.** `gender` and `SeniorCitizen` are model inputs.
  Check outcomes across groups before using scores in decisions that affect
  customers.
- **Population.** Scores may not transfer to other markets, products or time
  periods.
- **Dataset licence.** The data is the IBM Telco Customer Churn sample. Check
  its licence terms before redistributing it, for example in a public
  repository.

---

## Troubleshooting

| Message | Fix |
|---|---|
| `Split file not found … Run notebooks/02_data_preprocessing.py first` | Run notebook 02. |
| `Model artifact '…' not found … Run notebooks/03_model_training.py first` | Run notebook 03 (model files aren't in git). |
| `No final model selection … Run notebooks/04_model_evaluation.py first` | Run notebook 04. |
| `Saved split does not match the dataset` / `trained on a different split` | The data or split changed. Re-run notebooks 02 → 04. |
| `Model '…' was saved with scikit-learn X` warning | Your scikit-learn version differs from the training version. Install the pinned versions, or retrain with notebook 03. |
| `Schema validation failed: …` | The input has missing columns or invalid values; the message lists each problem. |
