import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, INTERNET_ADDON_COLUMNS, MODELS_DIR
from src.data_processing import apply_split, load_split, prepare_features_target
from src.explainability import (
    RETENTION_ACTIONS,
    TENURE_GROUP,
    historical_churn_rates,
    is_consistent,
    local_explanation,
    permutation_importance_table,
    recommend_actions,
    reference_profile,
    segment_action_summary,
    to_feature_frame,
    what_if,
)
from src.prediction import load_final_model


@pytest.fixture(scope="module")
def setup(raw_df):
    if not (MODELS_DIR / "final_model.json").is_file():
        pytest.skip("Run notebooks 03 and 04 to create the final model.")
    pipe, _, _ = load_final_model()
    X, y = prepare_features_target(raw_df)
    X_tr, X_te, y_tr, y_te = apply_split(X, y, load_split())
    return {
        "pipe": pipe,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_te": y_te,
        "ref": reference_profile(X_tr),
        "hist": historical_churn_rates(X_tr, y_tr),
    }


def first(X, **conditions):
    mask = np.ones(len(X), dtype=bool)
    for column, value in conditions.items():
        mask &= (X[column] == value).to_numpy()
    return X[mask].iloc[0]


def test_every_real_customer_is_consistent(raw_df):
    X, _ = prepare_features_target(raw_df)
    assert X.apply(is_consistent, axis=1).all()


def test_inconsistent_combinations_detected(raw_df):
    X, _ = prepare_features_target(raw_df)
    no_net = first(X, InternetService="No").copy()
    no_net["TechSupport"] = "Yes"
    assert not is_consistent(no_net)
    no_phone = first(X, PhoneService="No").copy()
    no_phone["MultipleLines"] = "Yes"
    assert not is_consistent(no_phone)


def test_to_feature_frame_restores_types(setup):
    row = setup["X_te"].iloc[0]  # an object Series
    frame = to_feature_frame([row])
    assert frame["tenure"].dtype == float and frame["SeniorCitizen"].dtype == int
    np.testing.assert_allclose(
        setup["pipe"].predict_proba(frame), setup["pipe"].predict_proba(setup["X_te"].iloc[[0]])
    )


def test_reference_profile_uses_training_data(setup):
    ref = setup["ref"]
    assert list(ref.index) == FEATURE_COLUMNS
    assert ref["tenure"] == setup["X_tr"]["tenure"].median()
    assert ref["Contract"] == setup["X_tr"]["Contract"].mode().iloc[0]


def test_local_explanation_effects_are_correct(setup):
    customer = setup["X_te"].iloc[0]
    expl = local_explanation(setup["pipe"], customer, setup["ref"])
    base = setup["pipe"].predict_proba(setup["X_te"].iloc[[0]])[:, 1][0]
    np.testing.assert_allclose(expl["effect"], base - expl["risk if typical"])
    assert "TotalCharges" not in set(expl["feature"])  # grouped with tenure
    # Only features that differ from the typical value are explained.
    shown = set(expl["feature"]) - {TENURE_GROUP}
    assert all(str(customer[f]) != str(setup["ref"][f]) for f in shown)


def test_local_explanation_never_creates_impossible_customers(setup):
    no_net = first(setup["X_te"], InternetService="No")
    expl = local_explanation(setup["pipe"], no_net, setup["ref"])
    # Switching InternetService alone (or one add-on alone) would be inconsistent.
    assert not set(expl["feature"]) & ({"InternetService"} | set(INTERNET_ADDON_COLUMNS))


def test_what_if_changes_only_one_feature(setup):
    X = setup["X_te"].head(20)
    same = what_if(setup["pipe"], X, "gender", X["gender"])
    np.testing.assert_allclose(same, setup["pipe"].predict_proba(X)[:, 1])


def test_recommendations_follow_both_rules(setup):
    customer = first(setup["X_te"], Contract="Month-to-month", InternetService="Fiber optic")
    recs = recommend_actions(setup["pipe"], customer, setup["hist"], min_reduction=0.0)
    assert (recs["model reduction"] >= 0).all()
    assert (recs["historical churn (proposed)"] < recs["historical churn (current)"]).all()
    assert recs["model reduction"].is_monotonic_decreasing


def test_recommendations_only_applicable_actions(setup):
    two_year = first(setup["X_te"], Contract="Two year")
    recs = recommend_actions(setup["pipe"], two_year, setup["hist"], min_reduction=0.0)
    assert not recs["change"].str.startswith("Contract").any()
    no_net = first(setup["X_te"], InternetService="No")
    recs = recommend_actions(setup["pipe"], no_net, setup["hist"], min_reduction=0.0)
    assert not recs["change"].str.split(":").str[0].isin(INTERNET_ADDON_COLUMNS).any()


def test_high_min_reduction_suppresses_everything(setup):
    customer = setup["X_te"].iloc[0]
    assert recommend_actions(setup["pipe"], customer, setup["hist"], min_reduction=1.0).empty


def test_segment_summary_counts(setup):
    X = setup["X_te"].head(200)
    summary = segment_action_summary(setup["pipe"], X, setup["hist"])
    assert (summary["recommended for"] <= summary["applicable customers"]).all()
    contract_1y = summary[summary["action"].str.contains("one-year")]
    assert contract_1y["applicable customers"].iloc[0] == (X["Contract"] == "Month-to-month").sum()
    assert segment_action_summary(setup["pipe"], X.iloc[0:0], setup["hist"]).empty


def test_actions_reference_valid_levels():
    from src.config import CATEGORY_LEVELS

    for act in RETENTION_ACTIONS:
        assert act.proposed in CATEGORY_LEVELS[act.feature]
        assert set(act.applies_to) <= CATEGORY_LEVELS[act.feature]
        assert act.proposed not in act.applies_to


def test_permutation_importance_shape(setup):
    X, y = setup["X_te"].head(300), setup["y_te"].head(300)
    table = permutation_importance_table(setup["pipe"], X, y, n_repeats=2)
    assert set(table["feature"]) == set(FEATURE_COLUMNS)
    assert table["importance"].is_monotonic_decreasing
