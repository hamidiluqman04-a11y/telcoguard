import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="TelcoGuard 05 - Churn insights")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.express as px

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.config import CATEGORY_LEVELS, FEATURE_COLUMNS, ID_COLUMN, MODELS_DIR, NUMERIC_COLUMNS, SPLIT_PATH
    from src.data_processing import (
        SchemaError,
        apply_split,
        load_raw_data,
        load_split,
        prepare_features_target,
    )
    from src.explainability import (
        TENURE_GROUP,
        historical_churn_rates,
        local_explanation,
        permutation_importance_table,
        recommend_actions,
        reference_profile,
        segment_action_summary,
    )
    from src.prediction import (
        RISK_LABELS,
        assign_risk_category,
        load_final_model,
        read_customer_csv,
        score_customers,
        upload_template,
    )

    PLOT_TEMPLATE = "plotly_white"
    RISK_COLORS = {"Low": "#54A24B", "Medium": "#F58518", "High": "#E45756"}
    SOURCE_TEST = "Test customers (not seen during training)"
    SOURCE_ALL = "All customers in the dataset"
    SOURCE_UPLOAD = "Uploaded CSV"
    TABLE_COLUMNS = [
        "churn_probability", "risk_category", "flagged", "tenure", "Contract", "MonthlyCharges",
        "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
    ]


@app.cell
def _():
    mo.md(r"""
    # TelcoGuard 05 — Churn insights

    An interactive tool for exploring **which customers are at risk of
    churning, why the model thinks so, and what retention actions might be
    worth testing**.

    Please read these two points before using the results:

    1. **Risk is an estimate, not a certainty.** A 70% churn probability means
       that among many similar customers, about 7 in 10 are expected to leave.
       This particular customer may well stay, and a 10% customer may leave.
    2. **Association is not causation.** The model learned patterns from
       historical data. "Customers on two-year contracts churn less" does not
       prove that moving someone to a two-year contract will keep them. The
       suggestions below are **hypotheses to test** (for example, with a
       controlled retention campaign), not guaranteed outcomes.
    """)
    return


@app.cell
def _():
    pipeline, selection, model_meta = load_final_model(MODELS_DIR)
    threshold = float(selection["threshold"])
    return model_meta, pipeline, selection, threshold


@app.cell
def _():
    raw_df = load_raw_data()
    _features, _target = prepare_features_target(raw_df)
    X_train, X_test, y_train, y_test = apply_split(_features, _target, load_split(SPLIT_PATH))
    reference = reference_profile(X_train)
    history = historical_churn_rates(X_train, y_train)
    return X_test, history, raw_df, reference, y_test


@app.cell
def _(model_meta, selection, threshold):
    _m = selection["test_metrics_at_threshold"]
    mo.md(f"""
    ## 1. The model behind these numbers

    | | |
    |---|---|
    | Model | **{selection['model_label']}** (full preprocessing + model pipeline) |
    | Chosen because | {selection['selection_rule']} |
    | Decision threshold | **{threshold:.2f}**: customers at or above this are *flagged* |
    | Test-set recall / precision at this threshold | **{_m['recall']:.1%}** of churners caught; **{_m['precision']:.1%}** of flagged customers actually churned |
    | Test-set ROC-AUC / PR-AUC | {_m['roc_auc']:.3f} / {_m['average_precision']:.3f} |
    | Trained with | scikit-learn {model_meta['versions']['scikit-learn']}, {model_meta['n_train']:,} training customers |

    So roughly **{1 - _m['precision']:.0%} of flagged customers would not have
    churned**, and about **{1 - _m['recall']:.0%} of churners are not flagged**.
    These figures come from Notebook 4.
    """)
    return


@app.cell
def _(raw_df):
    source = mo.ui.dropdown(
        options=[SOURCE_TEST, SOURCE_ALL, SOURCE_UPLOAD], value=SOURCE_TEST, label="Customers to score"
    )
    upload = mo.ui.file(filetypes=[".csv"], kind="area", label="Drop a customer CSV here or click to browse")
    template_download = mo.download(
        data=upload_template(raw_df.head(5)),
        filename="telcoguard_upload_template.csv",
        mimetype="text/csv",
        label="Download example CSV",
    )
    return source, template_download, upload


@app.cell
def _(source, template_download, upload):
    _schema = pd.DataFrame(
        [{"column": ID_COLUMN, "allowed values": "any unique text (optional)"}]
        + [{"column": c, "allowed values": "number ≥ 0" + (" (blank allowed)" if c == "TotalCharges" else "")} for c in NUMERIC_COLUMNS]
        + [{"column": c, "allowed values": ", ".join(sorted(map(str, v)))} for c, v in CATEGORY_LEVELS.items()]
        + [{"column": "Churn", "allowed values": "Yes / No (optional, used only to compare)"}]
    )
    mo.vstack(
        [
            mo.md("""
    ## 2. Choose customers to score

    - **Test customers** (default): the 20% held out from training. Their
      scores are honest out-of-sample estimates.
    - **All customers**: includes the 80% the model was trained on. Their
      scores are *optimistic*, because the model has seen these customers.
    - **Uploaded CSV**: your own customers in the same format as the original
      file. The file is validated before scoring, and any problem is reported.
    """),
            source,
            upload if source.value == SOURCE_UPLOAD else mo.md(""),
            mo.accordion(
                {
                    "Expected CSV format": mo.vstack(
                        [
                            mo.md("Column names must match exactly. Extra columns are ignored."),
                            template_download,
                            mo.ui.table(_schema, selection=None, page_size=25, show_download=False),
                        ]
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(X_test, pipeline, raw_df, source, threshold, upload):
    _error = None
    _raw = None
    if source.value == SOURCE_UPLOAD:
        mo.stop(
            not upload.value,
            mo.callout(mo.md("Upload a CSV above to score your own customers."), kind="info"),
        )
        try:
            _raw = read_customer_csv(upload.contents())
        except SchemaError as exc:
            _error = str(exc)
        source_note = f"customers from the uploaded file `{upload.name()}`"
    elif source.value == SOURCE_ALL:
        _raw = raw_df
        source_note = "customers from the whole dataset (⚠️ includes training customers, whose scores are optimistic)"
    else:
        _raw = raw_df.set_index(ID_COLUMN).loc[X_test.index].reset_index()
        source_note = "held-out test customers"

    if _raw is not None and _error is None:
        try:
            scored = score_customers(pipeline, _raw, threshold)
        except SchemaError as exc:
            _error = str(exc)
    mo.stop(
        _error is not None,
        mo.callout(
            mo.md(f"**The file was rejected, so nothing was scored.**\n\n```\n{_error}\n```\n\nFix the file and upload it again."),
            kind="danger",
        ),
    )
    return scored, source_note


@app.cell
def _(threshold):
    risk_cutoffs = mo.ui.range_slider(
        start=0.05,
        stop=0.95,
        step=0.01,
        value=[round(threshold, 2), 0.60],
        label="Risk cut-offs (Medium from · High from)",
        show_value=True,
        full_width=True,
    )
    return (risk_cutoffs,)


@app.cell
def _(risk_cutoffs, scored):
    _medium_from, _high_from = risk_cutoffs.value
    mo.stop(
        _medium_from >= _high_from,
        mo.callout("The two cut-offs must be different. Move the handles apart.", kind="warn"),
    )
    risk_df = scored.assign(
        risk_category=assign_risk_category(scored["churn_probability"], _medium_from, _high_from)
    )
    return (risk_df,)


@app.cell
def _(risk_cutoffs, risk_df, source_note, threshold):
    _counts = (
        risk_df["risk_category"].value_counts().reindex(RISK_LABELS).rename_axis("risk").reset_index(name="customers")
    )
    _hist = px.histogram(
        risk_df, x="churn_probability", color="risk_category", nbins=40,
        color_discrete_map=RISK_COLORS, category_orders={"risk_category": RISK_LABELS},
        template=PLOT_TEMPLATE, title="Distribution of estimated churn probability",
    )
    _hist.add_vline(x=threshold, line_dash="dash", annotation_text=f"model threshold {threshold:.2f}")
    _hist.update_layout(height=360, xaxis_tickformat=".0%", legend_title=None)
    _expected = risk_df["churn_probability"].sum()
    _has_truth = "actual_churn" in risk_df.columns
    _check = ""
    if _has_truth:
        _by_risk = risk_df.groupby("risk_category", observed=False).agg(
            customers=("actual_churn", "size"),
            mean_estimated_risk=("churn_probability", "mean"),
            actual_churn_rate=("actual_churn", "mean"),
        ).reset_index()
        _check = mo.vstack(
            [
                mo.md(f"""
    **Reality check (these customers have known outcomes):** the model
    expected **{_expected:,.0f}** churners; **{int(risk_df['actual_churn'].sum()):,}**
    actually churned. Within each category, compare average estimated risk with
    the actual churn rate:
    """),
                mo.ui.table(_by_risk.round(3), selection=None, show_download=False),
            ]
        )
    _lo, _hi = risk_cutoffs.value
    mo.vstack(
        [
            mo.md(f"""
    ## 3. Risk overview

    Scoring **{len(risk_df):,}** {source_note}.
    Categories: **Low** < {_lo:.0%} ≤ **Medium** < {_hi:.0%} ≤ **High**. The
    Medium cut-off starts at the model threshold, so Medium + High = flagged
    customers. Adjust the cut-offs to match your team's capacity.

    Summing the probabilities gives the **expected number of churners**:
    about **{_expected:,.0f}** ({_expected / len(risk_df):.1%}). We can't say
    *which* of them will leave, only how many are likely to.
    """),
            risk_cutoffs,
            mo.hstack(
                [mo.ui.table(_counts, selection=None, show_download=False), _hist], widths=[1, 3], align="center"
            ),
            _check,
        ]
    )
    return


@app.cell
def _(risk_df):
    risk_filter = mo.ui.multiselect(options=RISK_LABELS, value=["High", "Medium"], label="Risk")
    contract_filter = mo.ui.multiselect(
        options=sorted(risk_df["Contract"].unique()), value=sorted(risk_df["Contract"].unique()), label="Contract"
    )
    internet_filter = mo.ui.multiselect(
        options=sorted(risk_df["InternetService"].unique()),
        value=sorted(risk_df["InternetService"].unique()),
        label="Internet",
    )
    tenure_filter = mo.ui.range_slider(
        start=0, stop=max(72, int(risk_df["tenure"].max())), step=1,
        value=[0, max(72, int(risk_df["tenure"].max()))], label="Tenure (months)", show_value=True,
    )
    return contract_filter, internet_filter, risk_filter, tenure_filter


@app.cell
def _(contract_filter, internet_filter, risk_df, risk_filter, tenure_filter):
    _lo, _hi = tenure_filter.value
    filtered_df = risk_df[
        risk_df["risk_category"].isin(risk_filter.value)
        & risk_df["Contract"].isin(contract_filter.value)
        & risk_df["InternetService"].isin(internet_filter.value)
        & risk_df["tenure"].between(_lo, _hi)
    ].sort_values("churn_probability", ascending=False)
    _columns = TABLE_COLUMNS + (["actual_churn"] if "actual_churn" in risk_df.columns else [])
    customer_table = mo.ui.table(
        filtered_df[_columns].reset_index(),
        selection="single",
        page_size=10,
        format_mapping={"churn_probability": "{:.1%}".format, "MonthlyCharges": "{:.2f}".format},
        show_download=False,
        label="Click a row to explain that customer below (default: highest risk)",
    )
    return customer_table, filtered_df


@app.cell
def _(
    contract_filter,
    customer_table,
    filtered_df,
    internet_filter,
    risk_df,
    risk_filter,
    threshold,
    tenure_filter,
):
    _export = filtered_df.assign(decision_threshold=threshold).reset_index()
    _all_export = risk_df.assign(decision_threshold=threshold).reset_index()
    mo.vstack(
        [
            mo.md(f"""
    ## 4. Find customers

    Filter, search and sort (click a column header). **{len(filtered_df):,}**
    of {len(risk_df):,} customers match. The CSV downloads include every model
    feature, the probability, the risk category, the flag and the threshold
    used.
    """),
            mo.hstack([risk_filter, contract_filter, internet_filter], justify="start", wrap=True),
            tenure_filter,
            customer_table,
            mo.hstack(
                [
                    mo.download(
                        data=_export.to_csv(index=False).encode(),
                        filename="telcoguard_predictions_filtered.csv",
                        mimetype="text/csv",
                        label=f"Download filtered ({len(_export):,} rows)",
                    ),
                    mo.download(
                        data=_all_export.to_csv(index=False).encode(),
                        filename="telcoguard_predictions_all.csv",
                        mimetype="text/csv",
                        label=f"Download all scored ({len(_all_export):,} rows)",
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(X_test, pipeline, y_test):
    with mo.status.spinner(title="Computing permutation importance on test customers…"):
        importance = permutation_importance_table(pipeline, X_test, y_test)
    return (importance,)


@app.cell
def _(importance):
    _fig = px.bar(
        importance.sort_values("importance"), x="importance", y="feature", error_x="std", orientation="h",
        template=PLOT_TEMPLATE, title="Permutation importance (drop in test PR-AUC when the feature is shuffled)",
    )
    _fig.update_traces(marker_color="#4C78A8")
    _fig.update_layout(height=560, yaxis_title=None)
    _top = importance.head(3)["feature"].tolist()
    _useless = importance[importance["importance"] <= importance["std"]]["feature"].tolist()
    mo.vstack(
        [
            mo.md(f"""
    ## 5. What drives the model overall?

    **Permutation importance** shuffles one column at a time on the held-out
    test customers and measures how much the model's PR-AUC drops. A big drop
    means the model relies on that column.

    - The most important features are **{", ".join(_top)}**.
    - {len(_useless)} feature(s) have importance no larger than their own noise
      ({", ".join(_useless[:8])}{"…" if len(_useless) > 8 else ""}), so the model
      barely uses them.
    - **Caveat:** related features share credit. `tenure` and `TotalCharges` carry
      overlapping information, so each one alone looks less important than the
      pair. Importance describes the *model*, not what causes churn.
    """),
            _fig,
        ]
    )
    return


@app.cell
def _(customer_table, filtered_df):
    mo.stop(
        filtered_df.empty,
        mo.callout("No customers match the filters, so there is no one to explain.", kind="warn"),
    )
    _selected = customer_table.value
    selected_id = (
        _selected[ID_COLUMN].iloc[0] if _selected is not None and len(_selected) else filtered_df.index[0]
    )
    selected_customer = filtered_df.loc[selected_id]
    return selected_customer, selected_id


@app.cell
def _(pipeline, reference, selected_customer, selected_id, threshold):
    _p = float(selected_customer["churn_probability"])
    _expl = local_explanation(pipeline, selected_customer, reference)
    _up = _expl[_expl["effect"] > 0.005].head(3)
    _down = _expl[_expl["effect"] < -0.005].head(3)

    def _describe(frame):
        return (
            "; ".join(
                f"**{f}** = {v} (vs typical {t}, {e:+.1%})"
                for f, v, t, e in zip(
                    frame["feature"], frame["customer value"], frame["typical value"], frame["effect"]
                )
            )
            or "none of note"
        )

    _plot = _expl.head(10).iloc[::-1]
    _fig = px.bar(
        _plot, x="effect", y="feature", orientation="h",
        color=_plot["effect"] > 0, color_discrete_map={True: "#E45756", False: "#54A24B"},
        hover_data=["customer value", "typical value", "risk if typical"],
        template=PLOT_TEMPLATE,
        title=f"Why {selected_id}'s estimated risk differs from a typical customer",
    )
    _fig.update_layout(height=420, showlegend=False, xaxis_tickformat="+.0%", yaxis_title=None,
                       xaxis_title="change in risk caused by the customer's actual value")
    _truth = (
        f" Actual outcome (known for this dataset): **{'churned' if selected_customer['actual_churn'] == 1 else 'stayed'}**."
        if "actual_churn" in selected_customer.index
        else ""
    )
    mo.vstack(
        [
            mo.md(f"""
    ## 6. Explaining one customer: `{selected_id}`

    Estimated churn risk **{_p:.1%}** → **{selected_customer['risk_category']}** risk,
    {"**flagged**" if _p >= threshold else "not flagged"} at threshold {threshold:.2f}.{_truth}

    **How to read the chart:** for each feature we ask the model, *"what would
    this customer's risk be if only this feature had the most common value
    among training customers?"* Red bars mean the customer's actual value
    **raises** the estimate; green bars mean it **lowers** it. Features where the
    customer already has the typical value are not shown. `{TENURE_GROUP}` is
    changed jointly to keep the customer realistic.

    - Pushing risk **up**: {_describe(_up)}
    - Pulling risk **down**: {_describe(_down)}

    These are model explanations. Effects interact and do not add up to the
    total risk.
    """),
            _fig,
        ]
    )
    return


@app.cell
def _():
    min_reduction = mo.ui.slider(
        0.0, 0.2, step=0.01, value=0.02, label="Minimum modelled risk reduction to suggest an action",
        show_value=True,
    )
    return (min_reduction,)


@app.cell
def _(history, min_reduction, pipeline, selected_customer, selected_id):
    _actions = recommend_actions(pipeline, selected_customer, history, min_reduction.value)
    _risk = selected_customer["risk_category"]
    if _risk == "Low":
        _body = mo.callout(
            mo.md(
                f"`{selected_id}` is **low risk**. No retention spend is suggested; "
                "resources are better used on Medium/High-risk customers."
            ),
            kind="success",
        )
    elif _actions.empty:
        _body = mo.callout(
            mo.md(
                "No action meets the rule (model reduction ≥ the minimum **and** a lower "
                "historical churn rate). This customer may need a conversation rather than a standard offer."
            ),
            kind="info",
        )
    else:
        _shown = _actions.copy()
        for _c in ["risk now", "model risk after", "model reduction", "historical churn (current)", "historical churn (proposed)"]:
            _shown[_c] = _shown[_c].map("{:.1%}".format)
        _best = _actions.iloc[0]
        _body = mo.vstack(
            [
                mo.md(f"""
    Top suggestion: **{_best['action']}**. The model's estimate drops from
    {_best['risk now']:.1%} to {_best['model risk after']:.1%}, and historically
    {_best['historical churn (proposed)']:.1%} of the
    {_best['customers in proposed group']:,} training customers in the proposed
    group churned, versus {_best['historical churn (current)']:.1%} in the
    customer's current group.
    """),
                mo.ui.table(_shown, selection=None, show_download=False),
            ]
        )
    mo.vstack(
        [
            mo.md(r"""
    ## 7. Retention suggestions for this customer

    Each candidate action is a realistic lever (contract, payment method,
    protective add-ons). An action is suggested only if **two independent
    signals agree**:

    1. **Model what-if:** the model's estimated risk drops by at least the
       minimum below when only that attribute is changed (price held constant,
       i.e. the offer is assumed to be free or discounted).
    2. **Historical evidence:** in the training data, customers with the
       proposed value churned less than customers with the current value.

    **Neither signal proves the action will work.** Customers who *choose*
    two-year contracts or tech support differ in many ways from those who don't
    (self-selection). Treat these as candidates for a controlled test, and
    measure the real effect before rolling out widely.
    """),
            min_reduction,
            _body,
        ]
    )
    return


@app.cell
def _(filtered_df, history, min_reduction, pipeline, threshold):
    _flagged = filtered_df[filtered_df["churn_probability"] >= threshold]
    mo.stop(
        _flagged.empty,
        mo.md("## 8. Segment view\n\nNo flagged customers in the current filter."),
    )
    _summary = segment_action_summary(pipeline, _flagged[FEATURE_COLUMNS], history, min_reduction.value)
    _fig = px.bar(
        _summary.sort_values("recommended for"), x="recommended for", y="action", orientation="h",
        hover_data={"applicable customers": True, "mean model reduction": ":.1%"},
        template=PLOT_TEMPLATE, title=f"Suggested actions across {len(_flagged):,} flagged customers in the filter",
    )
    _fig.update_traces(marker_color="#F58518")
    _fig.update_layout(height=380, yaxis_title=None, xaxis_title="customers for whom the action is suggested")
    _shown = _summary.assign(**{"mean model reduction": _summary["mean model reduction"].map("{:.1%}".format)})
    mo.vstack(
        [
            mo.md(f"""
    ## 8. Segment view: which actions apply most often?

    Across the **{len(_flagged):,} flagged customers** in the current filter,
    this shows how many customers each action could apply to, for how many it
    meets the suggestion rule, and the average modelled risk reduction. Use it
    to decide which **experiments** to run first, not as a forecast of
    customers saved.

    Large modelled reductions (especially for contracts) mostly reflect how
    strongly that attribute *separated* churners from non-churners in the past.
    They do **not** mean that share of customers would actually be retained.
    Only a controlled test can measure that.
    """),
            _fig,
            mo.ui.table(_shown, selection=None, show_download=False),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 9. Limitations

    - **Snapshot data.** The dataset is one moment in time with no dates, so the
      model cannot account for seasonality or changes in pricing, products or
      competitors. Re-check performance on fresh data before relying on it.
    - **Probabilities are estimates.** They were checked for calibration in
      Notebook 4, but individual customers can and will defy them.
    - **Explanations describe the model, not the world.** What-if changes hold
      everything else fixed, which real customers rarely do (for example, a new
      contract usually comes with a different price).
    - **No fairness review yet.** Features such as `gender` and `SeniorCitizen`
      are model inputs. Before using scores for decisions that affect customers,
      check that outcomes are fair across groups.
    - **Uploaded data must match the original format and population.** The
      model has not seen other markets or products, and its scores may not
      transfer to them.
    """)
    return


if __name__ == "__main__":
    app.run()
