import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="TelcoGuard 02 - Preprocessing")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.base import clone

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.config import (
        CATEGORICAL_COLUMNS,
        FEATURE_COLUMNS,
        ID_COLUMN,
        NUMERIC_COLUMNS,
        RANDOM_STATE,
        SPLIT_PATH,
        TARGET_COLUMN,
        TEST_SIZE,
    )
    from src.data_processing import (
        apply_split,
        build_preprocessor,
        load_raw_data,
        load_split,
        make_train_test_split,
        prepare_features_target,
        save_split,
    )


@app.cell
def _():
    mo.md(r"""
    # TelcoGuard 02 — Preprocessing

    Machine-learning models need clean, numeric input. This notebook turns the
    raw customer table into model-ready data **without leaking information**
    from the test set or the target into training.

    The steps are:

    1. Clean `TotalCharges` and handle its missing values.
    2. Separate the features (inputs) from the target (`Churn`).
    3. Split customers into a training set and a held-out test set (stratified).
    4. Build a `ColumnTransformer` for numeric and categorical columns.
    5. Wrap it in a `Pipeline` so it is only ever fitted on training data.

    The reusable code lives in `src/data_processing.py`; this notebook explains
    and demonstrates it.
    """)
    return


@app.cell
def _():
    raw_df = load_raw_data()
    return (raw_df,)


@app.cell
def _(raw_df):
    features, target = prepare_features_target(raw_df)
    return features, target


@app.cell
def _(features, raw_df):
    _blank_ids = features.index[features["TotalCharges"].isna()]
    _raw_values = raw_df.set_index(ID_COLUMN).loc[_blank_ids, "TotalCharges"]
    _before_after = pd.DataFrame(
        {
            "raw TotalCharges": _raw_values.map(repr),
            "after conversion": features.loc[_blank_ids, "TotalCharges"],
            "tenure": features.loc[_blank_ids, "tenure"],
        }
    )
    mo.vstack(
        [
            mo.md(f"""
    ## 1. Cleaning `TotalCharges`

    In the raw file `TotalCharges` is text, because **{len(_blank_ids)}** cells
    contain a single space. We convert the column to numbers with a *strict*
    rule: blank cells become missing (`NaN`), but any other non-numeric text
    (for example `"12,5"`) raises an error instead of silently becoming missing.

    **Why not drop these rows?** All of them are customers with tenure 0.
    Dropping them would teach the model nothing about brand-new customers, and
    new customers still need to be scored in production.

    **Why fill with 0, and why inside the pipeline?** A tenure-0 customer has
    genuinely been charged nothing yet, so 0 is the correct value rather than a
    guess like the mean. Putting the imputer inside the pipeline means exactly
    the same rule is saved with the model and applied to any future data.
    """),
            mo.ui.table(_before_after.reset_index(), selection=None),
        ]
    )
    return


@app.cell
def _(features, raw_df, target):
    _excluded = [c for c in raw_df.columns if c not in FEATURE_COLUMNS]
    mo.md(f"""
    ## 2. Features and target

    - **Target `y`:** `{TARGET_COLUMN}` mapped to 1 (Yes) / 0 (No); mean
      **{target.mean():.4f}** = the churn rate.
    - **Features `X`:** {features.shape[1]} columns —
      {len(NUMERIC_COLUMNS)} numeric ({", ".join(NUMERIC_COLUMNS)}) and
      {len(CATEGORICAL_COLUMNS)} categorical.
    - **Excluded:** {", ".join(f"`{c}`" for c in _excluded)}.
      - `{ID_COLUMN}` is a random identifier. It carries no real information,
        but a flexible model could memorise it and look better than it is.
        It is kept only as the row **index**, so predictions can be traced
        back to customers.
      - `{TARGET_COLUMN}` is what we predict; including it would be the most
        direct form of leakage.

    No feature is derived from the target, and no feature describes events
    *after* churn, so there is no obvious target leakage in the inputs.
    """)
    return


@app.cell
def _(features, target):
    X_train, X_test, y_train, y_test = make_train_test_split(features, target)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train):
    split_path = save_split(X_train.index, X_test.index, SPLIT_PATH)
    return (split_path,)


@app.cell
def _(X_test, X_train, features, split_path, target, y_test, y_train):
    # Re-load the file we just wrote and confirm it reproduces the same split.
    _X_tr, _X_te, _, _ = apply_split(features, target, load_split(split_path))
    _roundtrip_ok = _X_tr.index.equals(X_train.index) and _X_te.index.equals(X_test.index)
    _no_overlap = X_train.index.intersection(X_test.index).empty
    _split_summary = pd.DataFrame(
        {
            "set": ["train", "test", "all"],
            "customers": [len(y_train), len(y_test), len(target)],
            "churners": [int(y_train.sum()), int(y_test.sum()), int(target.sum())],
            "churn rate": [y_train.mean(), y_test.mean(), target.mean()],
        }
    )
    _split_summary["churn rate"] = _split_summary["churn rate"].map("{:.2%}".format)
    mo.vstack(
        [
            mo.md(f"""
    ## 3. Stratified train/test split

    We hold back **{TEST_SIZE:.0%}** of customers as a **test set** that no
    model will see during training or tuning. It is used once, in Notebook 4, to
    estimate how the final model performs on new customers.

    - **Stratified:** both sets keep the same churn rate (see table). Without
      stratification, a random split could put noticeably more churners in one
      set by chance, which distorts both training and evaluation.
    - **Reproducible:** `random_state={RANDOM_STATE}`.
    - **Shared:** the customer IDs of each set are saved to
      `{split_path.relative_to(PROJECT_ROOT).as_posix()}`. Notebooks 3 and 4
      read this file, so every model is trained and evaluated on exactly the
      same customers.
    - Train and test overlap: **{"none" if _no_overlap else "OVERLAP FOUND"}**.
      Saved file reproduces the split: **{"yes" if _roundtrip_ok else "NO"}**.
    """),
            mo.ui.table(_split_summary, selection=None),
        ]
    )
    return


@app.cell
def _():
    preprocessor = build_preprocessor()
    return (preprocessor,)


@app.cell
def _(preprocessor):
    mo.vstack(
        [
            mo.md(r"""
    ## 4. The `ColumnTransformer`

    Different column types need different treatment:

    | Columns | Steps | Why |
    |---|---|---|
    | numeric (`tenure`, `MonthlyCharges`, `TotalCharges`) | fill missing with 0 → standardise | 0 is the true value for tenure-0 customers; standardising (mean 0, std 1) puts features on comparable scales, which matters for regularised Logistic Regression |
    | categorical (16 columns) | one-hot encode | models need numbers; one column per level avoids inventing a false order such as "DSL < Fiber" |

    `handle_unknown="ignore"` means a category never seen during training
    becomes all zeros instead of crashing. In practice, new files are validated
    against the known categories *before* prediction, so unknown values are
    reported to the user rather than silently ignored.

    Tree models (Random Forest, gradient boosting) don't need scaling, but it
    doesn't hurt them, so all models share one preprocessor and are compared
    on equal terms.
    """),
            mo.Html(preprocessor._repr_html_()),
        ]
    )
    return


@app.cell
def _(X_train, preprocessor):
    # Demonstration only: fit a *copy* on the training data to inspect the
    # output. The real fitting happens inside each model pipeline (Notebook 3).
    demo_preprocessor = clone(preprocessor).fit(X_train)
    encoded_names = list(demo_preprocessor.get_feature_names_out())
    return demo_preprocessor, encoded_names


@app.cell
def _(X_test, X_train, demo_preprocessor, encoded_names):
    _train_encoded = demo_preprocessor.transform(X_train)
    _test_encoded = demo_preprocessor.transform(X_test)
    _scaler = demo_preprocessor.named_transformers_["numeric"].named_steps["scale"]
    _scaler_table = pd.DataFrame(
        {
            "column": NUMERIC_COLUMNS,
            "mean learned from train": _scaler.mean_.round(3),
            "std learned from train": _scaler.scale_.round(3),
        }
    )
    _n_numeric = len(NUMERIC_COLUMNS)
    mo.vstack(
        [
            mo.md(f"""
    ### 4a. What the fitted preprocessor produces

    The {X_train.shape[1]} raw feature columns become **{len(encoded_names)}**
    numeric columns ({_n_numeric} scaled numeric + {len(encoded_names) - _n_numeric}
    one-hot). Training matrix: **{_train_encoded.shape}**; test matrix:
    **{_test_encoded.shape}**. Missing values after transformation:
    **{int(np.isnan(_train_encoded).sum() + np.isnan(_test_encoded).sum())}**.

    The scaling statistics below were learned **from the training set only**.
    The test set is transformed with these same numbers, never refitted.
    """),
            mo.ui.table(_scaler_table, selection=None),
        ]
    )
    return


@app.cell
def _(X_train):
    row_picker = mo.ui.number(
        start=0,
        stop=len(X_train) - 1,
        value=0,
        step=1,
        label="Training row to inspect",
    )
    return (row_picker,)


@app.cell
def _(X_train, demo_preprocessor, encoded_names, row_picker):
    _row = X_train.iloc[[row_picker.value]]
    _encoded = pd.Series(demo_preprocessor.transform(_row)[0], index=encoded_names)
    _active = _encoded[(_encoded != 0) | _encoded.index.str.startswith("numeric__")]
    mo.vstack(
        [
            mo.md("""
    ### 4b. Inspect one customer before and after preprocessing

    Pick a row to see its raw values and the non-zero columns it turns into.
    Each categorical value lights up exactly one one-hot column.
    """),
            row_picker,
            mo.hstack(
                [
                    mo.ui.table(
                        _row.T.reset_index().set_axis(["feature", "raw value"], axis=1).astype(str),
                        selection=None,
                        page_size=20,
                        label=f"Raw ({_row.index[0]})",
                    ),
                    mo.ui.table(
                        _active.round(3).reset_index().set_axis(["encoded column", "value"], axis=1),
                        selection=None,
                        page_size=20,
                        label="Encoded (non-zero)",
                    ),
                ],
                widths="equal",
            ),
        ]
    )
    return


@app.cell
def _(X_train, features):
    _scaler_all = features[NUMERIC_COLUMNS].fillna(0).mean()
    _scaler_train = X_train[NUMERIC_COLUMNS].fillna(0).mean()
    _diff = pd.DataFrame(
        {
            "mean on train only": _scaler_train.round(3),
            "mean on ALL data (leaky)": _scaler_all.round(3),
            "difference": (_scaler_all - _scaler_train).round(3),
        }
    ).reset_index(names="column")
    mo.vstack(
        [
            mo.md(r"""
    ## 5. Why a `Pipeline` prevents leakage

    **Data leakage** means information the model shouldn't have (for example,
    test-set statistics) sneaks into training, so evaluation looks better than
    real-world performance will be.

    A common mistake is to scale or encode the *whole* dataset first and split
    afterwards: the scaler's mean and standard deviation would then include
    test customers. The table shows that difference for this split. Here it is
    small, but the principle matters, and it becomes serious during
    **cross-validation**, where the preprocessing must be refitted inside every
    fold.

    The fix is to put preprocessing and model in one `Pipeline`:

    ```python
    Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
    ```

    - `pipeline.fit(X_train, y_train)` fits the imputer, scaler and encoder on
      training data only.
    - In cross-validation, scikit-learn refits the *whole* pipeline in every
      fold, so each validation fold is treated like unseen data.
    - The saved pipeline contains the preprocessing, so predictions on new
      customers use exactly the training-time rules.
    """),
            mo.ui.table(_diff, selection=None),
        ]
    )
    return


@app.cell
def _(X_test, X_train, encoded_names, split_path):
    mo.md(f"""
    ## 6. Summary

    | Decision | Choice | Reason |
    |---|---|---|
    | `TotalCharges` blanks | strict numeric conversion → NaN → impute **0** in pipeline | all are tenure-0 customers with no charges yet |
    | Excluded columns | `{ID_COLUMN}`, `{TARGET_COLUMN}` | identifier / target |
    | Split | stratified {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed {RANDOM_STATE} → {len(X_train):,} train / {len(X_test):,} test | keep churn rate equal; reproducible |
    | Numeric features | impute 0 + standardise | comparable scales |
    | Categorical features | one-hot, ignore unknown | no false ordering; robust |
    | Leakage protection | everything inside one `Pipeline`, fitted on train only | test data never influences training |

    Encoded feature count: **{len(encoded_names)}**. Split saved to
    `{split_path.relative_to(PROJECT_ROOT).as_posix()}`.

    **Next:** Notebook 3 trains and tunes the models with cross-validation on the
    training set only.
    """)
    return


if __name__ == "__main__":
    app.run()
