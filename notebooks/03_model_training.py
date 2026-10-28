import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="TelcoGuard 03 - Model training")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.config import (
        CV_FOLDS,
        MODELS_DIR,
        RANDOM_STATE,
        RAW_DATA_PATH,
        SPLIT_PATH,
    )
    from src.data_processing import (
        apply_split,
        file_sha256,
        load_raw_data,
        load_split,
        prepare_features_target,
    )
    from src.training import (
        CV_SCORING,
        REFIT_METRIC,
        baseline_cv_scores,
        best_cv_scores,
        fold_scores,
        load_model_artifact,
        model_specs,
        params_at_grid_edge,
        save_model_artifact,
        tune_model,
    )

    PLOT_TEMPLATE = "plotly_white"
    METRIC_LABELS = {
        "average_precision": "PR-AUC (average precision)",
        "roc_auc": "ROC-AUC",
        "f1": "F1 @ 0.5",
        "precision": "Precision @ 0.5",
        "recall": "Recall @ 0.5",
        "accuracy": "Accuracy @ 0.5",
    }


@app.cell
def _():
    mo.md(r"""
    # TelcoGuard 03 — Model training

    We train three different kinds of model and tune each one fairly:

    | Model | Idea |
    |---|---|
    | **Logistic Regression** | a weighted sum of features → probability; simple and interpretable |
    | **Random Forest** | many decision trees on random subsets, averaged |
    | **Histogram Gradient Boosting** | trees built one after another, each fixing the previous ones' errors (scikit-learn's fast gradient boosting) |

    **Fairness rules**

    - Every model uses the **same training customers** (the split saved by
      Notebook 2) and the **same preprocessing pipeline**.
    - Hyperparameters are tuned with **5-fold stratified cross-validation on
      the training set only**. The test set is not loaded into any training or
      tuning step in this notebook.
    - Tuning maximises **PR-AUC (average precision)**, which measures how well
      churners are ranked above non-churners and is not inflated by the 73%
      majority class the way accuracy is.
    """)
    return


@app.cell
def _():
    raw_df = load_raw_data()
    features, target = prepare_features_target(raw_df)
    split = load_split(SPLIT_PATH)
    # Only the training part is kept. The test part is discarded here on
    # purpose so it cannot influence training; Notebook 4 evaluates on it.
    X_train, _X_test_unused, y_train, _y_test_unused = apply_split(features, target, split)
    return X_train, split, y_train


@app.cell
def _(X_train, split, y_train):
    mo.md(f"""
    ## 1. Training data

    - Training customers: **{len(X_train):,}** of {len(split):,}
      ({len(X_train) / len(split):.0%}), churn rate **{y_train.mean():.2%}**.
    - Features: **{X_train.shape[1]}** raw columns (no `customerID`, no `Churn`).
    - Split file: `{SPLIT_PATH.relative_to(PROJECT_ROOT).as_posix()}`
      (written by Notebook 2).
    - Cross-validation: **{CV_FOLDS}-fold stratified**, shuffled
      (`random_state={RANDOM_STATE}`). In each fold the model is trained on 4/5 of the training
      set and scored on the remaining 1/5, so every training customer is used
      for validation exactly once.
    """)
    return


@app.cell
def _():
    specs = model_specs()
    _rows = [
        {
            "model": spec.label,
            "hyperparameter": name.removeprefix("model__"),
            "values tried": ", ".join(str(v) for v in values),
        }
        for spec in specs.values()
        for name, values in spec.param_grid.items()
    ]
    _n = {s.label: int(np.prod([len(v) for v in s.param_grid.values()])) for s in specs.values()}
    mo.vstack(
        [
            mo.md(f"""
    ## 2. Search spaces

    Small, sensible grids: {", ".join(f"{k}: **{v}** combinations" for k, v in _n.items())}.
    Each combination is evaluated with {CV_FOLDS} folds.

    `class_weight="balanced"` makes mistakes on the minority class (churners)
    cost more during training. It is tried as a hyperparameter rather than
    assumed, because the decision threshold (Notebook 4) is another way to
    trade precision for recall.
    """),
            mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=20),
        ]
    )
    return (specs,)


@app.cell
def _(X_train, y_train):
    baseline_scores = baseline_cv_scores(X_train, y_train)
    return (baseline_scores,)


@app.cell
def _(baseline_scores):
    mo.md(f"""
    ## 3. Baseline: a model that ignores the customer

    A "dummy" model that gives every customer the average churn probability
    scores, in cross-validation: PR-AUC **{baseline_scores['average_precision']:.3f}**
    (= the churn rate), ROC-AUC **{baseline_scores['roc_auc']:.3f}** (random
    ranking), recall **{baseline_scores['recall']:.3f}**, and accuracy
    **{baseline_scores['accuracy']:.3f}**.

    Its 73% accuracy while catching zero churners is the clearest
    demonstration of why accuracy is not used to choose models. Any real model
    must beat this baseline's PR-AUC and ROC-AUC.
    """)
    return


@app.cell
def _(X_train, specs, y_train):
    with mo.status.spinner(title="Tuning models with cross-validation…"):
        searches = {name: tune_model(spec, X_train, y_train) for name, spec in specs.items()}
    return (searches,)


@app.cell
def _(baseline_scores, searches, specs):
    _rows = []
    for _name, _search in searches.items():
        _scores = best_cv_scores(_search)
        _rows.append(
            {"model": specs[_name].label}
            | {METRIC_LABELS[m]: _scores[m]["mean"] for m in CV_SCORING}
            | {"PR-AUC std": _scores[REFIT_METRIC]["std"], "tuning time (s)": round(_search.fit_seconds_, 1)}
        )
    _rows.append(
        {"model": "Baseline (dummy)"}
        | {METRIC_LABELS[m]: baseline_scores[m] for m in CV_SCORING}
    )
    cv_comparison = (
        pd.DataFrame(_rows)
        .sort_values(METRIC_LABELS[REFIT_METRIC], ascending=False, ignore_index=True)
    )
    return (cv_comparison,)


@app.cell
def _(cv_comparison, searches, specs):
    _long = []
    for _name, _search in searches.items():
        _s = best_cv_scores(_search)
        for _m in ("average_precision", "roc_auc", "f1", "precision", "recall"):
            _long.append(
                {"model": specs[_name].label, "metric": METRIC_LABELS[_m], "mean": _s[_m]["mean"], "std": _s[_m]["std"]}
            )
    _fig = px.bar(
        pd.DataFrame(_long),
        x="metric",
        y="mean",
        color="model",
        error_y="std",
        barmode="group",
        template=PLOT_TEMPLATE,
        title=f"Cross-validated scores of each model's best settings (mean ± std over {CV_FOLDS} folds)",
    )
    _fig.update_layout(yaxis_range=[0, 1], yaxis_title="score", height=430)
    _models_only = cv_comparison[cv_comparison["model"] != "Baseline (dummy)"]
    _best, _worst = _models_only.iloc[0], _models_only.iloc[-1]
    _pr = METRIC_LABELS[REFIT_METRIC]
    _gap = _best[_pr] - _worst[_pr]
    _max_std = _models_only["PR-AUC std"].max()
    _verdict = (
        f"The gap between the best and worst model (**{_gap:.3f}**) is "
        f"{'smaller than' if _gap < _max_std else 'comparable to or larger than'} "
        f"the fold-to-fold variation (std up to **{_max_std:.3f}**), so "
        + (
            "the models are **practically tied** on this data. The ranking may not survive a different split."
            if _gap < _max_std
            else "the ranking is reasonably clear."
        )
    )
    mo.vstack(
        [
            mo.md(f"""
    ## 4. Cross-validation results

    All numbers are averages over the {CV_FOLDS} validation folds of the
    **training set**. Ranked by {_pr}: **{_best['model']}** first
    ({_best[_pr]:.3f}), **{_worst['model']}** last ({_worst[_pr]:.3f}).
    {_verdict}

    All three models clearly beat the dummy baseline (PR-AUC
    {cv_comparison.loc[cv_comparison['model'] == 'Baseline (dummy)', _pr].iloc[0]:.3f}).
    Precision, recall and F1 use the default 0.5 threshold. At that threshold
    the models find roughly half of the churners; Notebook 4 shows how
    lowering the threshold trades precision for recall.
    """),
            mo.ui.table(cv_comparison.round(4), selection=None),
            _fig,
        ]
    )
    return


@app.cell
def _():
    fold_metric = mo.ui.dropdown(
        options={v: k for k, v in METRIC_LABELS.items()},
        value=METRIC_LABELS[REFIT_METRIC],
        label="Metric",
    )
    return (fold_metric,)


@app.cell
def _(fold_metric, searches, specs):
    _rows = [
        {"model": specs[_name].label, "fold": _k + 1, "score": _v}
        for _name, _search in searches.items()
        for _k, _v in enumerate(fold_scores(_search, fold_metric.value))
    ]
    _fig = px.strip(
        pd.DataFrame(_rows),
        x="model",
        y="score",
        color="model",
        hover_data=["fold"],
        template=PLOT_TEMPLATE,
        title=f"{fold_metric.selected_key} in each validation fold",
    )
    _fig.update_traces(marker_size=11)
    _fig.update_layout(showlegend=False, height=380)
    mo.vstack(
        [
            mo.md("""
    ### 4a. Fold-by-fold variation

    Each dot is one validation fold. Overlapping clouds mean the difference
    between models is within normal sampling noise.
    """),
            fold_metric,
            _fig,
        ]
    )
    return


@app.cell
def _(searches, specs):
    _rows = []
    for _name, _search in searches.items():
        _s = best_cv_scores(_search)[REFIT_METRIC]
        _edge = params_at_grid_edge(_search)
        _rows.append(
            {
                "model": specs[_name].label,
                "best settings": ", ".join(
                    f"{k.removeprefix('model__')}={v}" for k, v in _search.best_params_.items()
                ),
                "train PR-AUC": round(_s["train_mean"], 4),
                "validation PR-AUC": round(_s["mean"], 4),
                "gap (overfit)": round(_s["train_mean"] - _s["mean"], 4),
                "on grid edge": ", ".join(e.removeprefix("model__") for e in _edge) or "—",
            }
        )
    best_params_table = pd.DataFrame(_rows)
    _largest_gap = best_params_table.loc[best_params_table["gap (overfit)"].idxmax()]
    _any_edge = (best_params_table["on grid edge"] != "—").any()
    mo.vstack(
        [
            mo.md(f"""
    ## 5. Chosen hyperparameters and overfitting check

    - **Overfitting:** "train PR-AUC" is measured on the folds the model was
      trained on. A large gap to the validation score means the model memorises
      training customers. The largest gap is **{_largest_gap['model']}**
      ({_largest_gap['gap (overfit)']:.3f}).
    - **Grid edges:** {"some chosen values sit at the edge of the range tried (column *on grid edge*). A wider search there *might* help slightly, but given the fold-to-fold noise above, large gains are unlikely." if _any_edge else "no chosen numeric value sits at the edge of its range, so the grids were wide enough."}
    """),
            mo.ui.table(best_params_table, selection=None),
        ]
    )
    return


@app.cell
def _(specs):
    grid_model = mo.ui.dropdown(
        options={s.label: n for n, s in specs.items()},
        value=next(iter(specs.values())).label,
        label="Model",
    )
    return (grid_model,)


@app.cell
def _(grid_model, searches):
    _res = searches[grid_model.value].cv_results_
    # Show parameter values as text so `None` stays visible (pandas would
    # otherwise display it as NaN).
    _grid = pd.DataFrame(
        [{k.removeprefix("model__"): str(v) for k, v in p.items()} for p in _res["params"]]
    )
    for _m in ("average_precision", "roc_auc", "recall", "precision"):
        _grid[f"CV {METRIC_LABELS[_m]}"] = np.round(_res[f"mean_test_{_m}"], 4)
    _grid.insert(0, "rank", _res[f"rank_test_{REFIT_METRIC}"])
    _grid = _grid.sort_values("rank", ignore_index=True)
    mo.vstack(
        [
            mo.md("""
    ### 5a. Every combination tried

    All candidates for one model, ranked by CV PR-AUC. Notice how
    `class_weight` mostly changes **recall vs precision** at the 0.5 threshold
    while barely moving PR-AUC or ROC-AUC. It shifts the probabilities, not
    the ranking of customers.
    """),
            grid_model,
            mo.ui.table(_grid, selection=None, page_size=10),
        ]
    )
    return


@app.cell
def _(X_train, searches, specs, y_train):
    _extra = {
        "trained_on": "training split only (test set untouched)",
        "n_train": int(len(X_train)),
        "train_churn_rate": float(y_train.mean()),
        "data_file": RAW_DATA_PATH.name,
        "data_sha256": file_sha256(RAW_DATA_PATH),
        "split_file": SPLIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_sha256": file_sha256(SPLIT_PATH),
    }
    saved_artifacts = {
        name: save_model_artifact(search, specs[name], _extra, MODELS_DIR)
        for name, search in searches.items()
    }
    _ranking = sorted(
        searches, key=lambda n: best_cv_scores(searches[n])[REFIT_METRIC]["mean"], reverse=True
    )
    training_summary_path = MODELS_DIR / "training_summary.json"
    _ = training_summary_path.write_text(
        json.dumps(
            {
                "selection_metric": REFIT_METRIC,
                "cv_ranking": _ranking,
                "cv_scores": {n: best_cv_scores(searches[n]) for n in _ranking},
                **_extra,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return saved_artifacts, training_summary_path


@app.cell
def _(X_train, saved_artifacts, searches):
    # Reload every saved pipeline and confirm it reproduces the in-memory model.
    _sample = X_train.head(200)
    _rows = []
    for _name in saved_artifacts:
        _pipe, _meta = load_model_artifact(_name, MODELS_DIR)
        _same = np.allclose(
            _pipe.predict_proba(_sample), searches[_name].best_estimator_.predict_proba(_sample)
        )
        _rows.append(
            {
                "model": _meta["model_label"],
                "file": saved_artifacts[_name][0].name,
                "size (KB)": round(saved_artifacts[_name][0].stat().st_size / 1024, 1),
                "metadata": saved_artifacts[_name][1].name,
                "reload reproduces predictions": "yes" if _same else "NO",
            }
        )
    reload_check = pd.DataFrame(_rows)
    return (reload_check,)


@app.cell
def _(reload_check, training_summary_path):
    mo.vstack(
        [
            mo.md(f"""
    ## 6. Saved artifacts

    Each model's **entire fitted pipeline** (imputer, scaler, one-hot encoder
    and tuned model, refitted on the full training set) is saved
    with `joblib` to `models/`. Next to it is a JSON file with the chosen
    hyperparameters, CV scores, feature list, random seed, library versions,
    and SHA-256 fingerprints of the data and split files. With these you can
    tell exactly what a model was trained on, and detect if the data changed.

    An overall summary is in
    `{training_summary_path.relative_to(PROJECT_ROOT).as_posix()}`.
    """),
            mo.ui.table(reload_check, selection=None),
        ]
    )
    return


@app.cell
def _(cv_comparison):
    _models_only = cv_comparison[cv_comparison["model"] != "Baseline (dummy)"]
    _pr = METRIC_LABELS[REFIT_METRIC]
    mo.md(f"""
    ## 7. Summary and next steps

    - Cross-validated ranking by {_pr}:
      {" → ".join(f"**{m}** ({s:.3f})" for m, s in zip(_models_only["model"], _models_only[_pr]))}.
    - All models are far above the dummy baseline, but close to each other.
    - **No test-set data was used.** Everything above comes from
      cross-validation inside the training set.

    **Next (Notebook 4):** evaluate all three saved pipelines on the held-out
    test set, compare precision, recall, F1, ROC-AUC, PR-AUC and confusion
    matrices, and study how the decision threshold changes the trade-off
    between catching churners and wasting retention effort.
    """)
    return


if __name__ == "__main__":
    app.run()
