import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="TelcoGuard 01 - Data exploration")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.express as px

    # Make the project's `src` package importable whether this notebook is
    # opened with `marimo edit`, served with `marimo run`, or run with `python`.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.config import (
        CATEGORICAL_COLUMNS,
        ID_COLUMN,
        INTERNET_ADDON_COLUMNS,
        NUMERIC_COLUMNS,
        RAW_DATA_PATH,
        TARGET_COLUMN,
    )
    from src.data_processing import (
        churn_rate_by,
        clean_total_charges,
        data_quality_summary,
        encode_target,
        load_raw_data,
    )

    CHURN_COLORS = {"No": "#4C78A8", "Yes": "#E45756"}
    PLOT_TEMPLATE = "plotly_white"


@app.cell
def _():
    mo.md(r"""
    # TelcoGuard 01 — Data exploration

    This notebook explores the **real** IBM Telco customer churn file before any
    modelling. The goal is to understand the data, find quality problems, and
    see which customer characteristics are *associated* with churn.

    > **Association is not causation.** A group with a high churn rate is not
    > proof that the characteristic *causes* churn. Many columns are related to
    > each other (for example, contract type and tenure).

    All numbers and interpretations below are computed from the file itself, so
    they update automatically if the data changes.
    """)
    return


@app.cell
def _():
    raw_df = load_raw_data()
    return (raw_df,)


@app.cell
def _(raw_df):
    mo.vstack(
        [
            mo.md(f"""
    ## 1. Dataset overview

    Loaded **{len(raw_df):,} rows × {raw_df.shape[1]} columns** from
    `data/{RAW_DATA_PATH.name}`. Each row is one customer. `{TARGET_COLUMN}`
    is the target we want to predict, and `{ID_COLUMN}` is an identifier only
    (it will never be used as a model feature).

    The schema was validated on load: all expected columns exist and every
    categorical column contains only known values.
    """),
            mo.ui.table(raw_df.head(10), selection=None, label="First 10 rows"),
        ]
    )
    return


@app.cell
def _(raw_df):
    quality_df = data_quality_summary(raw_df)
    n_duplicate_rows = int(raw_df.duplicated().sum())
    n_duplicate_ids = int(raw_df[ID_COLUMN].duplicated().sum())
    return n_duplicate_ids, n_duplicate_rows, quality_df


@app.cell
def _(n_duplicate_ids, n_duplicate_rows, quality_df):
    _nan_total = int(quality_df["missing (NaN)"].sum())
    _blank_cols = quality_df.loc[quality_df["blank text"] > 0, ["column", "blank text"]]
    _blank_text = (
        ", ".join(f"`{c}` ({n})" for c, n in zip(_blank_cols["column"], _blank_cols["blank text"]))
        if len(_blank_cols)
        else "none"
    )
    _tc_dtype = quality_df.set_index("column").loc["TotalCharges", "dtype"]
    mo.vstack(
        [
            mo.md(f"""
    ## 2. Data quality

    - **Missing values (NaN) as read by pandas:** {_nan_total}.
    - **Hidden blanks (cells containing only spaces):** {_blank_text}.
    - **Duplicate rows:** {n_duplicate_rows}; **duplicate customer IDs:** {n_duplicate_ids}.
    - `TotalCharges` was read as type **`{_tc_dtype}`** (text), not as a number.
      That is a warning sign: a numeric column is read as text when some cells
      are not valid numbers.

    So "0 missing values" is misleading on its own. The blanks are hidden as
    whitespace and must be found explicitly.
    """),
            mo.ui.table(quality_df, selection=None, label="Column summary"),
        ]
    )
    return


@app.cell
def _(raw_df):
    clean_df = clean_total_charges(raw_df)
    blank_total_rows = clean_df[clean_df["TotalCharges"].isna()]
    return blank_total_rows, clean_df


@app.cell
def _(blank_total_rows, clean_df):
    _n_blank = len(blank_total_rows)
    _all_new = bool((blank_total_rows["tenure"] == 0).all())
    _n_tenure0 = int((clean_df["tenure"] == 0).sum())
    _churned = int((blank_total_rows[TARGET_COLUMN] == "Yes").sum())
    _verdict = (
        f"**All {_n_blank} have `tenure == 0`**, and they are the only customers "
        f"with tenure 0 ({_n_tenure0} in total). These are brand-new customers "
        "who have not received a bill yet, so their true total charge is "
        "effectively **0**, not unknown."
        if _all_new and _n_blank == _n_tenure0
        else "**Not all of them are new customers**, so treating blanks as 0 "
        "would be wrong. This needs further investigation."
    )
    mo.vstack(
        [
            mo.md(f"""
    ### 2a. The blank `TotalCharges` values

    Converting `TotalCharges` to numbers turns **{_n_blank}** blank cells into
    missing values. {_verdict} {_churned} of them churned.

    **Decision for preprocessing (Notebook 2):** keep `TotalCharges` as a
    number with these values missing, and fill them with **0** inside the model
    pipeline, so the same rule is applied to any future data. We will not drop
    these rows, because new customers must also be scored.
    """),
            mo.ui.table(
                blank_total_rows[
                    [ID_COLUMN, "tenure", "MonthlyCharges", "TotalCharges", "Contract", TARGET_COLUMN]
                ],
                selection=None,
                label="Rows with blank TotalCharges",
            ),
        ]
    )
    return


@app.cell
def _(clean_df):
    numeric_summary = clean_df[NUMERIC_COLUMNS].describe().T
    _billed = clean_df[clean_df["tenure"] > 0]
    charge_ratio = _billed["TotalCharges"] / (_billed["tenure"] * _billed["MonthlyCharges"])
    tenure_total_corr = float(clean_df["tenure"].corr(clean_df["TotalCharges"]))
    n_negative = int((clean_df[NUMERIC_COLUMNS] < 0).sum().sum())
    return charge_ratio, n_negative, numeric_summary, tenure_total_corr


@app.cell
def _(charge_ratio, n_negative, numeric_summary, tenure_total_corr):
    _q05, _q50, _q95 = charge_ratio.quantile([0.05, 0.5, 0.95])
    mo.vstack(
        [
            mo.md(f"""
    ### 2b. Numeric ranges and invalid values

    - Negative values in numeric columns: **{n_negative}**.
    - `tenure` ranges from **{numeric_summary.loc['tenure', 'min']:.0f}** to
      **{numeric_summary.loc['tenure', 'max']:.0f}** months.
    - `MonthlyCharges` ranges from **{numeric_summary.loc['MonthlyCharges', 'min']:.2f}**
      to **{numeric_summary.loc['MonthlyCharges', 'max']:.2f}**.
    - **Consistency check:** for billed customers, `TotalCharges ÷ (tenure ×
      MonthlyCharges)` has a median of **{_q50:.3f}** (5th–95th percentile
      {_q05:.3f}–{_q95:.3f}). Values close to 1 mean the columns agree; small
      deviations are expected because prices change over time.
    - `tenure` and `TotalCharges` are strongly correlated (r = **{tenure_total_corr:.2f}**).
      `TotalCharges` is largely "tenure × price", which matters when we interpret
      model coefficients later.

    No impossible values were found.
    """),
            mo.ui.table(numeric_summary.round(2).reset_index(names="column"), selection=None),
        ]
    )
    return


@app.cell
def _(clean_df):
    category_counts = pd.concat(
        [
            clean_df[_col]
            .value_counts()
            .rename_axis("level")
            .reset_index(name="customers")
            .assign(column=_col)
            for _col in CATEGORICAL_COLUMNS
        ],
        ignore_index=True,
    )[["column", "level", "customers"]]
    category_counts["level"] = category_counts["level"].astype(str)
    category_counts["share"] = (category_counts["customers"] / len(clean_df)).round(4)
    return (category_counts,)


@app.cell
def _(category_counts):
    _smallest = category_counts.nsmallest(1, "customers").iloc[0]
    mo.vstack(
        [
            mo.md(f"""
    ## 3. Categorical distributions

    Every categorical column and how many customers fall into each level. The
    smallest group is `{_smallest['column']}` = "{_smallest['level']}" with
    **{_smallest['customers']:,}** customers ({_smallest['share']:.1%}), so no
    level is extremely rare.

    Note that **"No internet service"** and **"No phone service"** are real
    categories (customers who don't have that service), not missing data.
    `SeniorCitizen` is stored as 0/1, where 1 means senior.
    """),
            mo.ui.table(category_counts, selection=None, page_size=15),
        ]
    )
    return


@app.cell
def _(clean_df):
    churn_flag = encode_target(clean_df[TARGET_COLUMN])
    baseline_churn_rate = float(churn_flag.mean())
    # A display-friendly copy: SeniorCitizen as No/Yes and a 0/1 churn column.
    explore_df = clean_df.assign(
        SeniorCitizen=clean_df["SeniorCitizen"].map({0: "No", 1: "Yes"}),
        churned=churn_flag,
    )
    return baseline_churn_rate, churn_flag, explore_df


@app.cell
def _(baseline_churn_rate, churn_flag):
    _counts = (
        churn_flag.map({0: "No", 1: "Yes"})
        .value_counts()
        .rename_axis(TARGET_COLUMN)
        .reset_index(name="customers")
    )
    _fig = px.bar(
        _counts,
        x=TARGET_COLUMN,
        y="customers",
        color=TARGET_COLUMN,
        color_discrete_map=CHURN_COLORS,
        text="customers",
        template=PLOT_TEMPLATE,
        title="Customers by churn status",
    )
    _fig.update_layout(showlegend=False, height=350)
    _n_yes = int(churn_flag.sum())
    _n_no = int(len(churn_flag) - _n_yes)
    mo.vstack(
        [
            mo.md(f"""
    ## 4. Target: churn distribution

    **{_n_yes:,}** customers churned and **{_n_no:,}** stayed, so the
    **baseline churn rate is {baseline_churn_rate:.2%}**.

    The classes are **imbalanced** (about 1 churner for every
    {_n_no / _n_yes:.1f} non-churners). A useless model that predicts "no churn"
    for everyone would already be **{1 - baseline_churn_rate:.1%} accurate**,
    while catching zero churners. This is why later notebooks judge models on
    recall, precision, F1, ROC-AUC and PR-AUC instead of accuracy. It is also
    why the train/test split must be **stratified** (keep the same churn rate in
    both parts).
    """),
            _fig,
        ]
    )
    return


@app.cell
def _(baseline_churn_rate, explore_df):
    def rate_bar(column, title):
        rates = churn_rate_by(explore_df, column)
        fig = px.bar(
            rates,
            x=column,
            y="churn_rate",
            text=rates["churn_rate"].map("{:.1%}".format),
            hover_data={"customers": True, "churned": True, "churn_rate": ":.1%"},
            template=PLOT_TEMPLATE,
            title=title,
        )
        fig.add_hline(
            y=baseline_churn_rate,
            line_dash="dash",
            annotation_text=f"overall {baseline_churn_rate:.1%}",
        )
        fig.update_traces(marker_color="#E45756", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="churn rate", height=380)
        return rates, fig

    contract_rates, contract_fig = rate_bar("Contract", "Churn rate by contract type")
    payment_rates, payment_fig = rate_bar("PaymentMethod", "Churn rate by payment method")
    return contract_fig, contract_rates, payment_fig, payment_rates, rate_bar


@app.cell
def _(contract_fig, contract_rates, payment_fig, payment_rates):
    _c_hi, _c_lo = contract_rates.iloc[0], contract_rates.iloc[-1]
    _p_hi, _p_lo = payment_rates.iloc[0], payment_rates.iloc[-1]
    mo.vstack(
        [
            mo.md(f"""
    ## 5. Contract and payment method

    - **Contract:** "{_c_hi['Contract']}" customers churn at
      **{_c_hi['churn_rate']:.1%}** versus **{_c_lo['churn_rate']:.1%}** for
      "{_c_lo['Contract']}" — about **{_c_hi['churn_rate'] / _c_lo['churn_rate']:.0f}×**
      higher. "{_c_hi['Contract']}" is also the largest group
      ({_c_hi['share_of_customers']:.0%} of customers).
    - **Payment method:** "{_p_hi['PaymentMethod']}" has the highest churn rate
      (**{_p_hi['churn_rate']:.1%}**) and "{_p_lo['PaymentMethod']}" the lowest
      (**{_p_lo['churn_rate']:.1%}**).

    Caution: these are raw associations. For example, customers on long contracts
    may also have longer tenure, so the two effects overlap. The model in later
    notebooks considers all features together.
    """),
            contract_fig,
            payment_fig,
        ]
    )
    return


@app.cell
def _(explore_df):
    _long = explore_df.melt(
        id_vars=["churned"],
        value_vars=INTERNET_ADDON_COLUMNS,
        var_name="service",
        value_name="status",
    )
    service_rates = (
        _long.groupby(["service", "status"], observed=True)["churned"]
        .agg(customers="size", churn_rate="mean")
        .reset_index()
    )
    internet_rates = churn_rate_by(explore_df, "InternetService")
    return internet_rates, service_rates


@app.cell
def _(baseline_churn_rate, internet_rates, rate_bar, service_rates):
    _wide = service_rates.pivot(index="service", columns="status", values="churn_rate")
    _wide["gap"] = _wide["No"] - _wide["Yes"]
    _wide = _wide.sort_values("gap", ascending=False)
    _top, _bottom = _wide.iloc[0], _wide.iloc[-1]
    _no_internet = float(_wide["No internet service"].iloc[0])

    _service_fig = px.bar(
        service_rates,
        x="service",
        y="churn_rate",
        color="status",
        barmode="group",
        category_orders={
            "service": list(_wide.index),
            "status": ["No", "Yes", "No internet service"],
        },
        color_discrete_map={"No": "#E45756", "Yes": "#4C78A8", "No internet service": "#BAB0AC"},
        hover_data={"customers": True, "churn_rate": ":.1%"},
        template=PLOT_TEMPLATE,
        title="Churn rate by add-on service status",
    )
    _service_fig.add_hline(y=baseline_churn_rate, line_dash="dash")
    _service_fig.update_layout(yaxis_tickformat=".0%", yaxis_title="churn rate", height=420)
    _, _internet_fig = rate_bar("InternetService", "Churn rate by internet service")
    _i_hi, _i_lo = internet_rates.iloc[0], internet_rates.iloc[-1]

    mo.vstack(
        [
            mo.md(f"""
    ## 6. Service categories

    - **Internet service:** "{_i_hi['InternetService']}" customers churn at
      **{_i_hi['churn_rate']:.1%}**; InternetService = "{_i_lo['InternetService']}"
      at **{_i_lo['churn_rate']:.1%}**.
    - **Add-on services:** the biggest gap is for **{_top.name}**: customers
      *without* it churn at **{_top['No']:.1%}** versus **{_top['Yes']:.1%}**
      with it. The smallest gap is for **{_bottom.name}**
      ({_bottom['No']:.1%} vs {_bottom['Yes']:.1%}).
    - Customers with **no internet service** churn at only **{_no_internet:.1%}**
      (the same group appears in every add-on column).

    Protective-looking services (security, support) may partly reflect *who*
    buys them (for example, more engaged or longer-tenure customers) rather than
    an effect of the service itself.
    """),
            _internet_fig,
            _service_fig,
        ]
    )
    return


@app.cell
def _():
    category_picker = mo.ui.dropdown(
        options=CATEGORICAL_COLUMNS,
        value="Contract",
        label="Category",
    )
    split_picker = mo.ui.dropdown(
        options=["(none)"] + CATEGORICAL_COLUMNS,
        value="(none)",
        label="Split by",
    )
    return category_picker, split_picker


@app.cell
def _(baseline_churn_rate, category_picker, explore_df, split_picker):
    _col = category_picker.value
    _split = None if split_picker.value in ("(none)", _col) else split_picker.value
    _keys = [_col] + ([_split] if _split else [])
    _rates = (
        explore_df.groupby(_keys, observed=True)["churned"]
        .agg(customers="size", churn_rate="mean")
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )
    _fig = px.bar(
        _rates,
        x=_col,
        y="churn_rate",
        color=_split,
        barmode="group",
        hover_data={"customers": True, "churn_rate": ":.1%"},
        template=PLOT_TEMPLATE,
        title=f"Churn rate by {_col}" + (f", split by {_split}" if _split else ""),
    )
    _fig.add_hline(y=baseline_churn_rate, line_dash="dash", annotation_text="overall")
    _fig.update_layout(yaxis_tickformat=".0%", yaxis_title="churn rate", height=420)
    _hi, _lo = _rates.iloc[0], _rates.iloc[-1]
    _small = _rates[_rates["customers"] < 100]
    _label = lambda r: ", ".join(f"{k} = {r[k]}" for k in _keys)
    _warning = (
        f"\n\n⚠️ {len(_small)} group(s) have fewer than 100 customers, so their rates are noisy."
        if len(_small)
        else ""
    )
    mo.vstack(
        [
            mo.md("""
    ## 7. Interactive category explorer

    Pick any categorical column, and optionally split it by a second one, to
    see how churn rates vary. Hover over bars for customer counts.
    """),
            mo.hstack([category_picker, split_picker], justify="start"),
            _fig,
            mo.md(
                f"Highest: **{_label(_hi)}** at {_hi['churn_rate']:.1%} "
                f"({int(_hi['customers']):,} customers). Lowest: **{_label(_lo)}** at "
                f"{_lo['churn_rate']:.1%} ({int(_lo['customers']):,} customers).{_warning}"
            ),
        ]
    )
    return


@app.cell
def _(explore_df):
    tenure_bands = pd.cut(
        explore_df["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"],
        include_lowest=True,
    )
    tenure_band_rates = churn_rate_by(
        explore_df.assign(tenure_band=tenure_bands), "tenure_band"
    ).sort_values("tenure_band", ignore_index=True)
    return (tenure_band_rates,)


@app.cell
def _():
    tenure_bins = mo.ui.slider(5, 72, value=24, step=1, label="Histogram bins", show_value=True)
    tenure_norm = mo.ui.radio(
        options={"Customer count": None, "% within churn group": "percent"},
        value="% within churn group",
        label="Y axis",
        inline=True,
    )
    return tenure_bins, tenure_norm


@app.cell
def _(baseline_churn_rate, explore_df, tenure_band_rates, tenure_bins, tenure_norm):
    _hist = px.histogram(
        explore_df,
        x="tenure",
        color=TARGET_COLUMN,
        nbins=tenure_bins.value,
        histnorm=tenure_norm.value,
        barmode="overlay",
        opacity=0.65,
        color_discrete_map=CHURN_COLORS,
        template=PLOT_TEMPLATE,
        title="Tenure distribution by churn status",
    )
    _hist.update_layout(xaxis_title="tenure (months)", height=380)
    _bands = px.bar(
        tenure_band_rates,
        x="tenure_band",
        y="churn_rate",
        text=tenure_band_rates["churn_rate"].map("{:.1%}".format),
        hover_data={"customers": True},
        template=PLOT_TEMPLATE,
        title="Churn rate by tenure band (months)",
    )
    _bands.add_hline(y=baseline_churn_rate, line_dash="dash")
    _bands.update_traces(marker_color="#E45756", textposition="outside", cliponaxis=False)
    _bands.update_layout(yaxis_tickformat=".0%", yaxis_title="churn rate", height=380)
    _first, _last = tenure_band_rates.iloc[0], tenure_band_rates.iloc[-1]
    _med = explore_df.groupby(TARGET_COLUMN)["tenure"].median()
    mo.vstack(
        [
            mo.md(f"""
    ## 8. Tenure

    Customers in their **first year** ({_first['tenure_band']} months) churn at
    **{_first['churn_rate']:.1%}**, versus **{_last['churn_rate']:.1%}** for
    customers who have been around {_last['tenure_band']} months. The median
    tenure of churners is **{_med['Yes']:.0f} months**, compared with
    **{_med['No']:.0f} months** for those who stayed. Churn is concentrated
    early in the customer relationship.

    Note: this is a snapshot of current and recently-left customers, so
    long-tenure customers are by definition those who did not leave earlier
    (survivorship).
    """),
            mo.hstack([tenure_bins, tenure_norm], justify="start"),
            _hist,
            _bands,
        ]
    )
    return


@app.cell
def _(explore_df):
    charge_bands = pd.qcut(explore_df["MonthlyCharges"], q=4)
    charge_band_rates = churn_rate_by(
        explore_df.assign(
            charge_band=charge_bands.map(lambda iv: f"{iv.left:.0f}–{iv.right:.0f}")
        ),
        "charge_band",
    )
    charge_band_rates = charge_band_rates.sort_values(
        "charge_band", key=lambda s: s.str.split("–").str[0].astype(float), ignore_index=True
    )
    return (charge_band_rates,)


@app.cell
def _():
    charges_bins = mo.ui.slider(10, 100, value=40, step=5, label="Histogram bins", show_value=True)
    return (charges_bins,)


@app.cell
def _(baseline_churn_rate, charge_band_rates, charges_bins, explore_df):
    _hist = px.histogram(
        explore_df,
        x="MonthlyCharges",
        color=TARGET_COLUMN,
        nbins=charges_bins.value,
        histnorm="percent",
        barmode="overlay",
        opacity=0.65,
        color_discrete_map=CHURN_COLORS,
        template=PLOT_TEMPLATE,
        title="Monthly charges by churn status (% within group)",
    )
    _hist.update_layout(height=380)
    _box = px.box(
        explore_df,
        x=TARGET_COLUMN,
        y="MonthlyCharges",
        color=TARGET_COLUMN,
        color_discrete_map=CHURN_COLORS,
        template=PLOT_TEMPLATE,
        title="Monthly charges spread",
    )
    _box.update_layout(showlegend=False, height=380)
    _bands = px.bar(
        charge_band_rates,
        x="charge_band",
        y="churn_rate",
        text=charge_band_rates["churn_rate"].map("{:.1%}".format),
        hover_data={"customers": True},
        template=PLOT_TEMPLATE,
        title="Churn rate by monthly-charge quartile",
    )
    _bands.add_hline(y=baseline_churn_rate, line_dash="dash")
    _bands.update_traces(marker_color="#E45756", textposition="outside", cliponaxis=False)
    _bands.update_layout(yaxis_tickformat=".0%", yaxis_title="churn rate", height=380)
    _med = explore_df.groupby(TARGET_COLUMN)["MonthlyCharges"].median()
    _lowest_q, _highest_q = charge_band_rates.iloc[0], charge_band_rates.iloc[-1]
    _peak = charge_band_rates.loc[charge_band_rates["churn_rate"].idxmax()]
    mo.vstack(
        [
            mo.md(f"""
    ## 9. Monthly charges

    Churners have a higher median monthly charge (**{_med['Yes']:.2f}**) than
    customers who stayed (**{_med['No']:.2f}**). By quartile, churn is lowest in
    the cheapest band ({_lowest_q['charge_band']}: **{_lowest_q['churn_rate']:.1%}**)
    and highest in the {_peak['charge_band']} band (**{_peak['churn_rate']:.1%}**);
    the most expensive band ({_highest_q['charge_band']}) is at
    **{_highest_q['churn_rate']:.1%}**.

    Price is entangled with services: higher charges usually mean fibre internet
    and more add-ons, so this pattern overlaps with section 6.
    """),
            charges_bins,
            _hist,
            mo.hstack([_box, _bands], widths=[1, 2]),
        ]
    )
    return


@app.cell
def _(
    baseline_churn_rate,
    blank_total_rows,
    contract_rates,
    internet_rates,
    payment_rates,
    tenure_band_rates,
):
    mo.md(f"""
    ## 10. Summary and next steps

    **Data quality**

    - {len(blank_total_rows)} blank `TotalCharges` values, all for tenure-0
      customers → convert to numeric and fill with 0 in the pipeline.
    - No duplicates, no other missing values, no invalid categories or ranges.
    - `customerID` is a unique identifier → exclude from features.

    **Target**

    - Baseline churn rate **{baseline_churn_rate:.2%}** → imbalanced; use a
      stratified split and metrics beyond accuracy.

    **Strongest associations with churn (raw, one at a time)**

    - Contract: "{contract_rates.iloc[0]['Contract']}" {contract_rates.iloc[0]['churn_rate']:.1%}
      vs "{contract_rates.iloc[-1]['Contract']}" {contract_rates.iloc[-1]['churn_rate']:.1%}.
    - Tenure: first year {tenure_band_rates.iloc[0]['churn_rate']:.1%} vs
      longest-tenure band {tenure_band_rates.iloc[-1]['churn_rate']:.1%}.
    - Internet: "{internet_rates.iloc[0]['InternetService']}" {internet_rates.iloc[0]['churn_rate']:.1%}.
    - Payment: "{payment_rates.iloc[0]['PaymentMethod']}" {payment_rates.iloc[0]['churn_rate']:.1%}.

    These are correlations in historical data, not proven causes. Next,
    Notebook 2 turns these findings into a leakage-safe preprocessing pipeline.
    """)
    return


if __name__ == "__main__":
    app.run()
