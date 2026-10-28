import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, ID_COLUMN, MODELS_DIR, TARGET_COLUMN
from src.data_processing import SchemaError
from src.prediction import (
    assign_risk_category,
    load_final_model,
    read_customer_csv,
    score_customers,
    upload_template,
)


@pytest.fixture(scope="module")
def final_model():
    if not (MODELS_DIR / "final_model.json").is_file():
        pytest.skip("Run notebooks 03 and 04 to create the final model.")
    return load_final_model()


@pytest.fixture
def csv_bytes(raw_df):
    return raw_df.head(25).to_csv(index=False).encode()


# --- Reading uploads ---------------------------------------------------------


def test_read_valid_csv(csv_bytes):
    df = read_customer_csv(csv_bytes)
    assert len(df) == 25 and ID_COLUMN in df.columns


def test_read_handles_utf8_bom(csv_bytes):
    assert ID_COLUMN in read_customer_csv(b"\xef\xbb\xbf" + csv_bytes).columns


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "empty"),
        (b"   \n ", "empty"),
        (b"\xff\xfe\x00bad", "UTF-8"),
        (b"customerID,gender\n", "no customer rows"),
        (b'a,b\n1,"unterminated\n', "could not be read"),
    ],
)
def test_read_rejects_bad_files(content, message):
    with pytest.raises(SchemaError, match=message):
        read_customer_csv(content)


# --- Scoring -----------------------------------------------------------------


def test_score_returns_probabilities_and_flags(final_model, raw_df):
    pipe, sel, _ = final_model
    scored = score_customers(pipe, raw_df.head(50), sel["threshold"])
    assert scored.index.name == ID_COLUMN
    assert scored["churn_probability"].between(0, 1).all()
    assert (scored["flagged"] == (scored["churn_probability"] >= sel["threshold"])).all()
    assert "actual_churn" in scored.columns
    assert ID_COLUMN not in scored.columns  # ID is the index, never a feature


def test_score_matches_pipeline_directly(final_model, raw_df):
    from src.data_processing import prepare_features

    pipe, sel, _ = final_model
    scored = score_customers(pipe, raw_df.head(30), sel["threshold"])
    direct = pipe.predict_proba(prepare_features(raw_df.head(30)))[:, 1]
    np.testing.assert_allclose(scored["churn_probability"], direct)


def test_score_without_id_or_target(final_model, raw_df):
    pipe, sel, _ = final_model
    scored = score_customers(pipe, raw_df.head(3).drop(columns=[ID_COLUMN, TARGET_COLUMN]), sel["threshold"])
    assert list(scored.index) == ["row_1", "row_2", "row_3"]
    assert "actual_churn" not in scored.columns


def test_score_ignores_extra_columns_and_column_order(final_model, raw_df):
    pipe, sel, _ = final_model
    base = score_customers(pipe, raw_df.head(10), sel["threshold"])
    shuffled = raw_df.head(10)[list(reversed(raw_df.columns))].assign(notes="x")
    pd.testing.assert_series_equal(
        score_customers(pipe, shuffled, sel["threshold"])["churn_probability"], base["churn_probability"]
    )


def test_score_handles_blank_total_charges(final_model, raw_df):
    pipe, sel, _ = final_model
    new_customers = raw_df[raw_df["tenure"] == 0].head(3)
    assert score_customers(pipe, new_customers, sel["threshold"])["churn_probability"].notna().all()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.drop(columns=["Contract"]), "Missing required column"),
        (lambda d: d.assign(Contract="Weekly"), "unexpected value"),
        (lambda d: d.assign(tenure=-3), "negative"),
        (lambda d: d.assign(customerID="same-id"), "Duplicate"),
        (lambda d: d.assign(Churn="Maybe"), "Churn|Target"),
    ],
)
def test_score_rejects_invalid_input(final_model, raw_df, mutate, message):
    pipe, sel, _ = final_model
    with pytest.raises(SchemaError, match=message):
        score_customers(pipe, mutate(raw_df.head(10).copy()), sel["threshold"])


def test_score_rejects_bad_threshold(final_model, raw_df):
    pipe, _, _ = final_model
    with pytest.raises(ValueError):
        score_customers(pipe, raw_df.head(2), 1.5)


def test_template_roundtrip(final_model, raw_df):
    pipe, sel, _ = final_model
    template = read_customer_csv(upload_template(raw_df.head(5)))
    assert TARGET_COLUMN not in template.columns
    assert set(FEATURE_COLUMNS) <= set(template.columns)
    assert len(score_customers(pipe, template, sel["threshold"])) == 5


# --- Risk categories ---------------------------------------------------------


def test_risk_categories_boundaries():
    cats = assign_risk_category([0.1, 0.3, 0.59, 0.6, 0.99], 0.3, 0.6)
    assert cats.tolist() == ["Low", "Medium", "Medium", "High", "High"]
    assert cats.cat.ordered


@pytest.mark.parametrize("cuts", [(0.6, 0.3), (0.3, 0.3), (0.0, 0.5), (0.5, 1.0)])
def test_risk_categories_reject_bad_cutoffs(cuts):
    with pytest.raises(ValueError):
        assign_risk_category([0.5], *cuts)
