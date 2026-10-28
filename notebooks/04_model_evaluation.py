import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="TelcoGuard 04 - Model evaluation")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.config import MODELS_DIR, REPORTS_DIR, SPLIT_PATH
    from src.data_processing import (
        apply_split,
        file_sha256,
        load_raw_data,
        load_split,
        prepare_features_target,
    )
    from src.evaluation import (
        DEFAULT_THRESHOLD,
        bootstrap_ci,
        classification_metrics,
        confusion_counts,
        expected_cost,
        out_of_fold_proba,
        save_final_selection,
        threshold_for_recall,
        threshold_max_f1,
        threshold_min_cost,
        threshold_table,
    )
    from src.training import load_model_artifact

    PLOT_TEMPLATE = "plotly_white"
    # Decided before looking at test results (see section 1).
    FINAL_THRESHOLD_POLICY = "max_f1_oof"
    RECALL_TARGET = 0.80


@app.cell
def _():
    mo.md(r"""
    # TelcoGuard 04 — Model evaluation

    This is the first and only notebook that uses the **held-out test set**
    (20% of customers that no model has seen). The goal is an honest estimate of
    how the models will perform on new customers.

    To keep the test set honest, these rules were fixed *before* looking at it:

    | Decision | How it is made | Uses test set? |
    |---|---|---|
    | Which model is "final" | best cross-validated PR-AUC in Notebook 3 | ❌ no |
    | Decision threshold | chosen on **out-of-fold** predictions of the training set | ❌ no |
    | Reported performance | measured on the test set, once | ✅ only to *report* |

    The interactive threshold and cost explorers below also run on the training
    out-of-fold predictions, so moving a slider can never "tune on the test set".
    """)
    return


@app.cell
def _():
    raw_df = load_raw_data()
    features, target = prepare_features_target(raw_df)
    X_train, X_test, y_train, y_test = apply_split(features, target, load_split(SPLIT_PATH))
    return X_test, X_train, y_test, y_train


@app.cell
def _():
    training_summary = json.loads((MODELS_DIR / "training_summary.json").read_text(encoding="utf-8"))
    model_names = training_summary["cv_ranking"]
    _loaded = {name: load_model_artifact(name, MODELS_DIR) for name in model_names}
    pipelines = {name: pipe for name, (pipe, _meta) in _loaded.items()}
    metadata = {name: meta for name, (_pipe, meta) in _loaded.items()}
    labels = {name: metadata[name]["model_label"] for name in model_names}
    selected_name = model_names[0]

    _split_hash = file_sha256(SPLIT_PATH)
    _stale = [n for n in model_names if metadata[n]["split_sha256"] != _split_hash]
    if _stale:
        raise RuntimeError(
            f"Models {_stale} were trained on a different split. Re-run Notebooks 2 and 3."
        )
    return labels, metadata, model_names, pipelines, selected_name


@app.cell
def _(X_test, labels, metadata, model_names, selected_name, y_test):
    mo.md(f"""
    ## 1. Setup

    - Test customers: **{len(X_test):,}**, of whom **{int(y_test.sum())}** churned
      ({y_test.mean():.2%}).
    - Loaded {len(model_names)} saved pipelines; all were trained on the current
      split (fingerprint check passed).
    - **Final model (pre-registered):** **{labels[selected_name]}**, because it had
      the best cross-validated PR-AUC
      ({metadata[selected_name]['cv_scores']['average_precision']['mean']:.3f}) in Notebook 3.
      All three models are still evaluated below for transparency, but test
      results do **not** change this choice. Picking whichever model happens to
      score highest on the test set would make the test score optimistic.
    """)
    return


@app.cell
def _(X_test, model_names, pipelines):
    test_proba = {
        name: pd.Series(pipelines[name].predict_proba(X_test)[:, 1], index=X_test.index)
        for name in model_names
    }
    return (test_proba,)


@app.cell
def _(labels, metadata, model_names, test_proba, y_test, y_train):
    _rows = []
    for _name in model_names:
        _m = classification_metrics(y_test, test_proba[_name], DEFAULT_THRESHOLD)
        _ap_lo, _ap_hi = bootstrap_ci(y_test, test_proba[_name], average_precision_score)
        _roc_lo, _roc_hi = bootstrap_ci(y_test, test_proba[_name], roc_auc_score)
        _cv = metadata[_name]["cv_scores"]
        _rows.append(
            {
                "model": labels[_name],
                "PR-AUC": _m["average_precision"],
                "PR-AUC 95% CI": f"{_ap_lo:.3f}–{_ap_hi:.3f}",
                "CV PR-AUC": _cv["average_precision"]["mean"],
                "ROC-AUC": _m["roc_auc"],
                "ROC-AUC 95% CI": f"{_roc_lo:.3f}–{_roc_hi:.3f}",
                "CV ROC-AUC": _cv["roc_auc"]["mean"],
                "precision @0.5": _m["precision"],
                "recall @0.5": _m["recall"],
                "F1 @0.5": _m["f1"],
                "accuracy @0.5": _m["accuracy"],
            }
        )
    # Baseline: give everyone the training churn rate.
    _base = classification_metrics(y_test, np.full(len(y_test), y_train.mean()), DEFAULT_THRESHOLD)
    _rows.append(
        {
            "model": "Baseline (constant)",
            "PR-AUC": _base["average_precision"],
            "ROC-AUC": _base["roc_auc"],
            "precision @0.5": _base["precision"],
            "recall @0.5": _base["recall"],
            "F1 @0.5": _base["f1"],
            "accuracy @0.5": _base["accuracy"],
        }
    )
    test_metrics_default = pd.DataFrame(_rows)
    return (test_metrics_default,)


@app.cell
def _(labels, metadata, model_names, selected_name, test_metrics_default):
    _models = test_metrics_default.iloc[:-1]
    _sel = _models[_models["model"] == labels[selected_name]].iloc[0]
    _top = _models.loc[_models["PR-AUC"].idxmax()]
    _drift = (_models["PR-AUC"] - _models["CV PR-AUC"]).abs().max()
    _cv_std = max(metadata[n]["cv_scores"]["average_precision"]["std"] for n in model_names)
    _drift_text = (
        f"which is within about two fold standard deviations (2 × {_cv_std:.3f}), so the "
        "cross-validation estimates were realistic and there is no sign the models overfit "
        "the training data."
        if _drift <= 2 * _cv_std
        else f"which is more than two fold standard deviations (2 × {_cv_std:.3f}). The CV "
        "estimates were somewhat optimistic; treat the test score as the more realistic one."
    )
    _bounds = _models["PR-AUC 95% CI"].str.split("–", expand=True).astype(float)
    _overlap = _bounds[1].min() >= _bounds[0].max()
    _overlap_text = (
        "All three 95% intervals overlap, so the test set cannot reliably separate the "
        "models either, which matches the cross-validation result."
        if _overlap
        else "Not all 95% intervals overlap, so at least one model is distinguishably "
        "different on the test set."
    )
    _base = test_metrics_default.iloc[-1]
    mo.vstack(
        [
            mo.md(f"""
    ## 2. Test-set results (default threshold 0.5)

    - **{labels[selected_name]}** (final model): PR-AUC **{_sel['PR-AUC']:.3f}**
      (95% CI {_sel['PR-AUC 95% CI']}), ROC-AUC **{_sel['ROC-AUC']:.3f}**
      (95% CI {_sel['ROC-AUC 95% CI']}).
    - Highest test PR-AUC: **{_top['model']}** ({_top['PR-AUC']:.3f}).
      {_overlap_text}
    - Test scores differ from cross-validation scores by at most
      **{_drift:.3f}** PR-AUC, {_drift_text}
    - The constant baseline reaches **{_base['accuracy @0.5']:.1%} accuracy** with
      recall **{_base['recall @0.5']:.0%}**. Accuracy is listed for reference
      only and is **not** used to choose anything.

    The 95% intervals come from 1,000 bootstrap resamples of the test set. They
    show how much a score could change with a different sample of the same
    number of customers. Treat differences smaller than the interval width as
    noise.
    """),
            mo.ui.table(test_metrics_default.round(4), selection=None),
        ]
    )
    return


@app.cell
def _(labels, model_names, test_proba, y_test):
    _roc = go.Figure()
    _pr = go.Figure()
    for _name in model_names:
        _fpr, _tpr, _ = roc_curve(y_test, test_proba[_name])
        _p, _r, _ = precision_recall_curve(y_test, test_proba[_name])
        _auc = roc_auc_score(y_test, test_proba[_name])
        _ap = average_precision_score(y_test, test_proba[_name])
        _roc.add_scatter(x=_fpr, y=_tpr, mode="lines", name=f"{labels[_name]} ({_auc:.3f})")
        _pr.add_scatter(x=_r, y=_p, mode="lines", name=f"{labels[_name]} ({_ap:.3f})")
    _roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", line_color="grey", name="random (0.5)")
    _pr.add_hline(y=y_test.mean(), line_dash="dash", line_color="grey",
                  annotation_text=f"baseline = churn rate {y_test.mean():.2f}")
    _roc.update_layout(template=PLOT_TEMPLATE, title="ROC curve (test set)", xaxis_title="false positive rate",
                       yaxis_title="true positive rate (recall)", height=420, legend=dict(x=0.45, y=0.05))
    _pr.update_layout(template=PLOT_TEMPLATE, title="Precision–recall curve (test set)", xaxis_title="recall",
                      yaxis_title="precision", height=420, yaxis_range=[0, 1.02], legend=dict(x=0.45, y=0.95))
    mo.vstack(
        [
            mo.md(r"""
    ### 2a. ROC and precision–recall curves

    Each curve shows performance across **all possible thresholds**.

    - **ROC-AUC**: the probability that a random churner gets a higher score
      than a random non-churner. 0.5 = guessing, 1.0 = perfect.
    - **PR-AUC (average precision)**: how precise the model stays as we try to
      catch more churners. Its baseline is the churn rate, not 0.5, which makes
      it more informative when the positive class is the minority.
    """),
            _roc,
            _pr,
        ]
    )
    return


@app.cell
def _():
    def confusion_figure(counts: dict, title: str):
        z = [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]]
        fig = px.imshow(
            z,
            x=["stay", "churn"],
            y=["stayed", "churned"],
            labels=dict(x="predicted", y="actual", color="customers"),
            text_auto=True,
            color_continuous_scale="Blues",
            template=PLOT_TEMPLATE,
            title=title,
        )
        fig.update_traces(textfont_size=16)
        fig.update_layout(
            coloraxis_showscale=False, height=320, margin=dict(l=10, r=10, t=50, b=10), title_font_size=14
        )
        return fig

    return (confusion_figure,)


@app.cell
def _(confusion_figure, labels, model_names, selected_name, test_proba, y_test):
    _figs = [
        confusion_figure(confusion_counts(y_test, test_proba[n], DEFAULT_THRESHOLD), labels[n])
        for n in model_names
    ]
    _c = confusion_counts(y_test, test_proba[selected_name], DEFAULT_THRESHOLD)
    mo.vstack(
        [
            mo.md(f"""
    ## 3. Confusion matrices at threshold 0.5

    A customer is **flagged** as "will churn" when the model's probability is
    at least 0.5. For the final model ({labels[selected_name]}) on the
    {len(y_test):,} test customers:

    | | What happened | Count | Business meaning |
    |---|---|---|---|
    | **True positive** | flagged, and churned | **{_c['tp']}** | a churner we can try to retain |
    | **False positive** | flagged, but stayed | **{_c['fp']}** | retention effort spent on someone who wasn't leaving |
    | **False negative** | not flagged, but churned | **{_c['fn']}** | a churner we missed entirely |
    | **True negative** | not flagged, and stayed | **{_c['tn']}** | correctly left alone |

    At 0.5 the model misses **{_c['fn']} of {_c['fn'] + _c['tp']}** churners
    ({_c['fn'] / (_c['fn'] + _c['tp']):.0%}). In churn prevention a missed
    churner (lost revenue) is usually more expensive than an unnecessary
    retention offer, which is a strong reason to look at **lower thresholds**
    (section 5).
    """),
            mo.hstack(_figs, widths="equal"),
        ]
    )
    return


@app.cell
def _(labels, selected_name, test_metrics_default, test_proba, y_test):
    _base_acc = test_metrics_default.iloc[-1]["accuracy @0.5"]
    _churner_p = test_proba[selected_name][y_test == 1]
    _below = float((_churner_p < 0.5).mean())
    _mid = float(((_churner_p >= 0.3) & (_churner_p < 0.5)).mean())
    mo.md(f"""
    ## 4. Class imbalance

    Only **{y_test.mean():.1%}** of test customers churned. That has three
    consequences:

    1. **Accuracy is misleading.** Predicting "nobody churns" is
       {_base_acc:.1%} accurate and completely useless.
    2. **Precision and recall pull against each other.** Flagging more customers
       catches more churners (recall ↑) but includes more customers who would
       have stayed (precision ↓).
    3. **0.5 is not a natural cut-off.** With a minority class, many real
       churners get probabilities below 0.5. For {labels[selected_name]},
       **{_below:.0%}** of test churners score below 0.5, and **{_mid:.0%}**
       fall between 0.3 and 0.5, all missed at the default threshold.

    We handled imbalance with a **stratified** split and folds, metrics that
    focus on the churn class (PR-AUC, recall, precision, F1), `class_weight`
    as a tunable option, and an explicit **threshold choice**, instead of
    resampling the data.
    """)
    return


@app.cell
def _(X_train, model_names, pipelines, y_train):
    with mo.status.spinner(title="Computing out-of-fold training predictions…"):
        oof_proba = {name: out_of_fold_proba(pipelines[name], X_train, y_train) for name in model_names}
    return (oof_proba,)


@app.cell
def _(labels, model_names, oof_proba, test_proba, y_test, y_train):
    _rows = []
    for _name in model_names:
        _policies = {
            "default 0.5": DEFAULT_THRESHOLD,
            "max F1 (OOF)": threshold_max_f1(y_train, oof_proba[_name]),
            f"recall ≥ {RECALL_TARGET:.0%} (OOF)": threshold_for_recall(y_train, oof_proba[_name], RECALL_TARGET),
        }
        for _policy, _t in _policies.items():
            _m = classification_metrics(y_test, test_proba[_name], _t)
            _rows.append(
                {
                    "model": labels[_name],
                    "policy": _policy,
                    "threshold": _t,
                    "test precision": _m["precision"],
                    "test recall": _m["recall"],
                    "test F1": _m["f1"],
                    "share flagged": _m["flagged_share"],
                    "TP": _m["tp"],
                    "FP": _m["fp"],
                    "FN": _m["fn"],
                }
            )
    threshold_results = pd.DataFrame(_rows)
    return (threshold_results,)


@app.cell
def _(labels, selected_name, threshold_results):
    _sel = threshold_results[threshold_results["model"] == labels[selected_name]].set_index("policy")
    _d, _f = _sel.loc["default 0.5"], _sel.loc["max F1 (OOF)"]
    _r = _sel.iloc[2]
    mo.vstack(
        [
            mo.md(f"""
    ## 5. Choosing a decision threshold

    Each threshold is **chosen on the training out-of-fold predictions** and then
    **applied once** to the test set:

    - **max F1 (OOF):** the threshold with the best balance of precision and
      recall on training data.
    - **recall ≥ {RECALL_TARGET:.0%} (OOF):** the highest threshold that catches
      at least {RECALL_TARGET:.0%} of churners in training data. This suits a
      business that mainly wants to avoid missing churners.

    For the final model ({labels[selected_name]}) on the test set:

    | Policy | Threshold | Precision | Recall | Customers flagged |
    |---|---|---|---|---|
    | default | {_d['threshold']:.2f} | {_d['test precision']:.1%} | {_d['test recall']:.1%} | {_d['share flagged']:.1%} |
    | max F1 | {_f['threshold']:.2f} | {_f['test precision']:.1%} | {_f['test recall']:.1%} | {_f['share flagged']:.1%} |
    | recall ≥ {RECALL_TARGET:.0%} | {_r['threshold']:.2f} | {_r['test precision']:.1%} | {_r['test recall']:.1%} | {_r['share flagged']:.1%} |

    Lowering the threshold from 0.5 to {_f['threshold']:.2f} raises recall from
    {_d['test recall']:.0%} to {_f['test recall']:.0%} (FN {int(_d['FN'])} → {int(_f['FN'])})
    at the cost of more false positives (FP {int(_d['FP'])} → {int(_f['FP'])}).
    There is no universally "correct" threshold; it depends on the cost of a
    retention action versus the value of a saved customer (section 5b).
    """),
            mo.ui.table(threshold_results.round(3), selection=None, page_size=9),
        ]
    )
    return


@app.cell
def _(labels, model_names, selected_name):
    explore_model = mo.ui.dropdown(
        options={labels[n]: n for n in model_names}, value=labels[selected_name], label="Model"
    )
    explore_threshold = mo.ui.slider(0.05, 0.95, step=0.01, value=0.5, label="Threshold", show_value=True)
    return explore_model, explore_threshold


@app.cell
def _(confusion_figure, explore_model, explore_threshold, oof_proba, y_train):
    _proba = oof_proba[explore_model.value]
    _table = threshold_table(y_train, _proba)
    _long = _table.melt(
        id_vars="threshold", value_vars=["precision", "recall", "f1", "flagged_share"], var_name="metric"
    )
    _fig = px.line(
        _long, x="threshold", y="value", color="metric", template=PLOT_TEMPLATE,
        title="Training out-of-fold metrics vs threshold",
    )
    _fig.add_vline(x=explore_threshold.value, line_dash="dash")
    _fig.update_layout(
        yaxis_range=[0, 1.02], height=400, legend=dict(orientation="h", y=-0.25, title=None)
    )
    _m = classification_metrics(y_train, _proba, explore_threshold.value)
    mo.vstack(
        [
            mo.md("""
    ### 5a. Threshold explorer (training out-of-fold data)

    Move the slider to see the trade-off. These numbers are from the
    **training** set's out-of-fold predictions, not the test set, so exploring
    here does not bias the test results.
    """),
            mo.hstack([explore_model, explore_threshold], justify="start"),
            mo.hstack(
                [
                    _fig,
                    mo.vstack(
                        [
                            confusion_figure(_m, f"OOF confusion matrix @ {explore_threshold.value:.2f}"),
                            mo.md(
                                f"Precision **{_m['precision']:.1%}** · Recall **{_m['recall']:.1%}** · "
                                f"F1 **{_m['f1']:.3f}** · Flagged **{_m['flagged_share']:.1%}** of customers"
                            ),
                        ]
                    ),
                ],
                widths=[3, 2],
            ),
        ]
    )
    return


@app.cell
def _():
    contact_cost = mo.ui.number(start=0, stop=10_000, step=5, value=50, label="Cost per retention contact")
    churn_loss = mo.ui.number(start=0, stop=100_000, step=50, value=500, label="Value lost per churner")
    save_rate = mo.ui.slider(0.0, 1.0, step=0.05, value=0.3, label="Share of contacted churners saved", show_value=True)
    return churn_loss, contact_cost, save_rate


@app.cell
def _(
    churn_loss,
    contact_cost,
    labels,
    oof_proba,
    save_rate,
    selected_name,
    test_proba,
    y_test,
    y_train,
):
    _args = dict(contact_cost=contact_cost.value, churn_loss=churn_loss.value, save_rate=save_rate.value)
    _oof = oof_proba[selected_name]
    _t_best = threshold_min_cost(y_train, _oof, **_args)
    _curve = pd.DataFrame(
        {"threshold": np.round(np.arange(0.05, 0.951, 0.01), 2)}
    ).assign(cost_per_customer=lambda d: [expected_cost(y_train, _oof, t, **_args) / len(y_train) for t in d["threshold"]])
    _fig = px.line(_curve, x="threshold", y="cost_per_customer", template=PLOT_TEMPLATE,
                   title=f"Expected cost per customer vs threshold ({labels[selected_name]}, training OOF)")
    _fig.add_vline(x=_t_best, line_dash="dash", annotation_text=f"min cost @ {_t_best:.2f}")
    _fig.update_layout(height=360, yaxis_title="cost per customer")
    _test = {
        "Contact nobody": expected_cost(y_test, test_proba[selected_name], 1.01, **_args),
        "Contact everyone": expected_cost(y_test, test_proba[selected_name], 0.0, **_args),
        f"Model @ {_t_best:.2f}": expected_cost(y_test, test_proba[selected_name], _t_best, **_args),
    }
    _best_simple = min(_test["Contact nobody"], _test["Contact everyone"])
    _saving = _best_simple - _test[f"Model @ {_t_best:.2f}"]
    mo.vstack(
        [
            mo.md(r"""
    ### 5b. Cost-based threshold (hypothetical economics)

    The right threshold depends on money, not on F1. Enter **your own
    assumptions** below. The defaults are made-up placeholders, **not**
    estimates from this dataset, which contains no cost or revenue data
    about retention campaigns.

    - Flagging a customer costs the contact cost.
    - A flagged churner is saved with the given probability.
    - An unflagged churner is lost.
    """),
            mo.hstack([contact_cost, churn_loss, save_rate], justify="start"),
            _fig,
            mo.md(
                f"Under these assumptions the cheapest threshold on training data is **{_t_best:.2f}**. "
                f"Applied to the {len(y_test):,} test customers, the total cost is "
                + " · ".join(f"{k}: **{v:,.0f}**" for k, v in _test.items())
                + f". The model saves **{_saving:,.0f}** compared with the better simple strategy."
                + ("" if _saving > 0 else " ⚠️ With these assumptions, using the model does not pay off.")
            ),
        ]
    )
    return


@app.cell
def _(labels, model_names, test_proba, y_test):
    _fig = go.Figure()
    _rows = []
    for _name in model_names:
        _true, _pred = calibration_curve(y_test, test_proba[_name], n_bins=10, strategy="quantile")
        _fig.add_scatter(x=_pred, y=_true, mode="lines+markers", name=labels[_name])
        _rows.append(
            {"model": labels[_name], "Brier score": classification_metrics(y_test, test_proba[_name])["brier"]}
        )
    _fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", line_color="grey", name="perfect")
    _fig.update_layout(template=PLOT_TEMPLATE, title="Calibration (test set, 10 equal-size bins)",
                       xaxis_title="average predicted probability", yaxis_title="actual churn rate", height=420)
    _brier = pd.DataFrame(_rows)
    mo.vstack(
        [
            mo.md(f"""
    ## 6. Are the probabilities trustworthy? (calibration)

    Notebook 5 shows churn *probabilities*, so they should mean what they say:
    among customers given ~30%, about 30% should churn. Points close to the
    dashed diagonal mean good calibration. The **Brier score** (lower is
    better; the constant baseline scores {y_test.mean() * (1 - y_test.mean()):.3f})
    summarises probability accuracy.

    Even a well-calibrated probability is an **estimate for a group of similar
    customers**, not a certainty about one person: a customer at 70% risk may
    well stay, and one at 10% may leave.
    """),
            _fig,
            mo.ui.table(_brier.round(4), selection=None),
        ]
    )
    return


@app.cell
def _(labels, metadata, oof_proba, selected_name, test_proba, threshold_results, y_test, y_train):
    final_threshold = threshold_max_f1(y_train, oof_proba[selected_name])
    _test_m = classification_metrics(y_test, test_proba[selected_name], final_threshold)
    final_selection = {
        "model_name": selected_name,
        "model_label": labels[selected_name],
        "model_file": f"{selected_name}.joblib",
        "selection_rule": "best cross-validated PR-AUC (Notebook 3), fixed before test evaluation",
        "threshold": final_threshold,
        "threshold_policy": FINAL_THRESHOLD_POLICY,
        "threshold_rule": "maximise F1 on 5-fold out-of-fold training predictions",
        "test_metrics_at_threshold": _test_m,
        "test_metrics_at_0_5": classification_metrics(y_test, test_proba[selected_name], DEFAULT_THRESHOLD),
        "cv_scores": metadata[selected_name]["cv_scores"],
        "n_test": int(len(y_test)),
        "split_sha256": file_sha256(SPLIT_PATH),
    }
    final_selection_path = save_final_selection(final_selection)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_report_path = REPORTS_DIR / "test_threshold_results.csv"
    threshold_results.to_csv(evaluation_report_path, index=False)
    return evaluation_report_path, final_selection, final_selection_path


@app.cell
def _(evaluation_report_path, final_selection, final_selection_path):
    _m = final_selection["test_metrics_at_threshold"]
    mo.md(f"""
    ## 7. Final decision and summary

    **Final model:** {final_selection['model_label']} with threshold
    **{final_selection['threshold']:.2f}** (max F1 on training out-of-fold
    predictions). On the held-out test set:

    | Metric | Value |
    |---|---|
    | ROC-AUC | {_m['roc_auc']:.3f} |
    | PR-AUC | {_m['average_precision']:.3f} |
    | Precision | {_m['precision']:.3f} |
    | Recall | {_m['recall']:.3f} |
    | F1 | {_m['f1']:.3f} |
    | Customers flagged | {_m['flagged_share']:.1%} |
    | Confusion (TP / FP / FN / TN) | {_m['tp']} / {_m['fp']} / {_m['fn']} / {_m['tn']} |

    Saved to `{final_selection_path.relative_to(PROJECT_ROOT).as_posix()}` (used by
    Notebook 5) and `{evaluation_report_path.relative_to(PROJECT_ROOT).as_posix()}`.

    **Limitations**

    - The three models perform almost identically, so the "final model" choice
      matters little. Logistic Regression would be a reasonable alternative if
      simplicity and interpretability are preferred.
    - These are scores on one dataset snapshot of {final_selection['n_test']:,} test customers; the
      confidence intervals show the remaining uncertainty.
    - The threshold should be revisited with real campaign costs and save rates.
    - Good prediction ≠ knowing what *causes* churn or what will *prevent* it.
      That requires experiments (e.g. A/B-tested retention offers).
    """)
    return


if __name__ == "__main__":
    app.run()
