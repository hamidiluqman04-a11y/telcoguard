"""Model explanations and evidence-based retention suggestions.

Two kinds of explanation, both model-agnostic so they work for any pipeline:

- Global: permutation importance (how much a held-out score drops when one
  raw feature is shuffled).
- Local: "what would the predicted risk be if this one feature had a typical
  value?" for a single customer.

All "what-if" numbers describe the *model's* behaviour. They are not causal
effects: changing a customer's contract will not necessarily change their
behaviour the way the model's estimate changes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    INTERNET_ADDON_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
)

# Features that move together and must be changed jointly in explanations.
TENURE_GROUP = "tenure (+ TotalCharges)"


def permutation_importance_table(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "average_precision",
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Drop in `scoring` when each raw feature is shuffled (mean ± std)."""
    result = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def to_feature_frame(rows) -> pd.DataFrame:
    """Build a correctly typed feature frame from customer Series.

    A single row taken out of a DataFrame becomes an `object` Series, so the
    types are restored here before the pipeline sees the data.
    """
    frame = pd.DataFrame([r[FEATURE_COLUMNS] for r in rows])[FEATURE_COLUMNS]
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].astype(float)
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    text_columns = [c for c in CATEGORICAL_COLUMNS if c != "SeniorCitizen"]
    frame[text_columns] = frame[text_columns].astype(str)
    return frame


def is_consistent(row: pd.Series) -> bool:
    """Reject combinations that cannot exist in this data, e.g. an
    internet add-on for a customer without internet service."""
    no_internet = row["InternetService"] == "No"
    addons_none = all(row[c] == "No internet service" for c in INTERNET_ADDON_COLUMNS)
    addons_any_none = any(row[c] == "No internet service" for c in INTERNET_ADDON_COLUMNS)
    if no_internet != addons_none or (not no_internet and addons_any_none):
        return False
    no_phone = row["PhoneService"] == "No"
    return no_phone == (row["MultipleLines"] == "No phone service")


def reference_profile(X_train: pd.DataFrame) -> pd.Series:
    """A 'typical' customer: median of numeric and mode of categorical features
    in the training data."""
    values = {c: float(X_train[c].median()) for c in NUMERIC_COLUMNS}
    values |= {c: X_train[c].mode().iloc[0] for c in CATEGORICAL_COLUMNS}
    return pd.Series(values)[FEATURE_COLUMNS]


def _with_tenure(row: pd.Series, tenure: float) -> pd.Series:
    """Change tenure and keep TotalCharges plausible (tenure × monthly)."""
    changed = row.copy()
    changed["tenure"] = tenure
    changed["TotalCharges"] = tenure * row["MonthlyCharges"]
    return changed


def local_explanation(pipeline, customer: pd.Series, reference: pd.Series) -> pd.DataFrame:
    """For each feature, the predicted risk if *only that feature* took the
    reference value. `effect` > 0 means the customer's actual value pushes
    risk up compared with a typical value.

    Changes that would create impossible customers (see `is_consistent`) are
    skipped. Features interact, so the effects do not add up to the total.
    """
    customer = customer[FEATURE_COLUMNS]
    base = float(pipeline.predict_proba(to_feature_frame([customer]))[:, 1][0])
    variants, rows = [], []
    for feature in FEATURE_COLUMNS:
        if feature == "TotalCharges":
            continue  # handled together with tenure
        if feature == "tenure":
            label, changed = TENURE_GROUP, _with_tenure(customer, reference["tenure"])
            current, typical = customer["tenure"], reference["tenure"]
        else:
            label, changed = feature, customer.copy()
            changed[feature] = reference[feature]
            current, typical = customer[feature], reference[feature]
        if current == typical:
            continue
        if not is_consistent(changed):
            continue
        variants.append(changed)
        rows.append({"feature": label, "customer value": current, "typical value": typical})
    if not rows:
        return pd.DataFrame(columns=["feature", "customer value", "typical value", "risk if typical", "effect"])
    with_typical = pipeline.predict_proba(to_feature_frame(variants))[:, 1]
    table = pd.DataFrame(rows)
    table["risk if typical"] = with_typical
    table["effect"] = base - with_typical
    table["customer value"] = table["customer value"].astype(str)
    table["typical value"] = table["typical value"].astype(str)
    return table.reindex(table["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)


@dataclass(frozen=True)
class RetentionAction:
    feature: str
    proposed: object
    applies_to: tuple
    action: str


# Levers a retention team can plausibly offer. Price is held constant in the
# what-if, i.e. each scenario assumes the offer comes at no extra charge.
RETENTION_ACTIONS = [
    RetentionAction("Contract", "One year", ("Month-to-month",), "Offer a one-year contract (e.g. with a loyalty discount)"),
    RetentionAction("Contract", "Two year", ("Month-to-month", "One year"), "Offer a two-year contract (e.g. with a loyalty discount)"),
    RetentionAction("PaymentMethod", "Credit card (automatic)", ("Electronic check", "Mailed check"), "Encourage automatic payment (credit card)"),
    RetentionAction("PaymentMethod", "Bank transfer (automatic)", ("Electronic check", "Mailed check"), "Encourage automatic payment (bank transfer)"),
    RetentionAction("OnlineSecurity", "Yes", ("No",), "Offer the online-security add-on (e.g. free trial)"),
    RetentionAction("TechSupport", "Yes", ("No",), "Offer tech support (e.g. free trial)"),
    RetentionAction("OnlineBackup", "Yes", ("No",), "Offer the online-backup add-on (e.g. free trial)"),
    RetentionAction("DeviceProtection", "Yes", ("No",), "Offer device protection (e.g. free trial)"),
]


def historical_churn_rates(X: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame]:
    """Observed churn rate and customer count per level of each action feature."""
    out = {}
    for feature in {a.feature for a in RETENTION_ACTIONS}:
        out[feature] = (
            pd.DataFrame({"level": X[feature], "churn": y.to_numpy()})
            .groupby("level")["churn"]
            .agg(churn_rate="mean", customers="size")
        )
    return out


def what_if(pipeline, X: pd.DataFrame, feature: str, value) -> np.ndarray:
    """Predicted risk for every row of X with one feature set to `value`."""
    changed = X[FEATURE_COLUMNS].copy()
    changed[feature] = value
    return pipeline.predict_proba(changed)[:, 1]


def recommend_actions(
    pipeline,
    customer: pd.Series,
    history: dict[str, pd.DataFrame],
    min_reduction: float = 0.02,
) -> pd.DataFrame:
    """Applicable retention actions for one customer, ranked by the model's
    estimated risk reduction, with the historical churn rates as evidence.

    An action is only listed if the model estimates at least `min_reduction`
    lower risk **and** the historical churn rate of the proposed level is
    lower than the customer's current level (both signals must agree).
    """
    row = to_feature_frame([customer])
    risk_now = float(pipeline.predict_proba(row)[:, 1][0])
    records = []
    for act in RETENTION_ACTIONS:
        current = customer[act.feature]
        if current not in act.applies_to:
            continue
        risk_after = float(what_if(pipeline, row, act.feature, act.proposed)[0])
        rates = history[act.feature]
        hist_now = float(rates.loc[current, "churn_rate"]) if current in rates.index else np.nan
        hist_after = float(rates.loc[act.proposed, "churn_rate"]) if act.proposed in rates.index else np.nan
        reduction = risk_now - risk_after
        if reduction >= min_reduction and hist_after < hist_now:
            records.append(
                {
                    "action": act.action,
                    "change": f"{act.feature}: {current} → {act.proposed}",
                    "risk now": risk_now,
                    "model risk after": risk_after,
                    "model reduction": reduction,
                    "historical churn (current)": hist_now,
                    "historical churn (proposed)": hist_after,
                    "customers in proposed group": int(rates.loc[act.proposed, "customers"]),
                }
            )
    columns = [
        "action", "change", "risk now", "model risk after", "model reduction",
        "historical churn (current)", "historical churn (proposed)", "customers in proposed group",
    ]
    return (
        pd.DataFrame(records, columns=columns)
        .sort_values("model reduction", ascending=False, ignore_index=True)
    )


def segment_action_summary(
    pipeline,
    X: pd.DataFrame,
    history: dict[str, pd.DataFrame],
    min_reduction: float = 0.02,
) -> pd.DataFrame:
    """For a group of customers: how many each action applies to, how many
    meet the recommendation rule, and the average modelled reduction."""
    if X.empty:
        return pd.DataFrame(
            columns=["action", "applicable customers", "recommended for", "mean model reduction"]
        )
    base = pipeline.predict_proba(X[FEATURE_COLUMNS])[:, 1]
    rows = []
    for act in RETENTION_ACTIONS:
        mask = X[act.feature].isin(act.applies_to).to_numpy()
        if not mask.any():
            continue
        after = what_if(pipeline, X[mask], act.feature, act.proposed)
        reduction = base[mask] - after
        rates = history[act.feature]["churn_rate"]
        hist_ok = X.loc[mask, act.feature].map(rates).to_numpy() > rates.get(act.proposed, np.inf)
        recommended = (reduction >= min_reduction) & hist_ok
        rows.append(
            {
                "action": act.action,
                "applicable customers": int(mask.sum()),
                "recommended for": int(recommended.sum()),
                "mean model reduction": float(reduction.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("recommended for", ascending=False, ignore_index=True)
