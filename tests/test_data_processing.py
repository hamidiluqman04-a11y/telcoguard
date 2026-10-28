import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, ID_COLUMN, RAW_COLUMNS, TARGET_COLUMN
from src.data_processing import (
    SchemaError,
    churn_rate_by,
    clean_total_charges,
    data_quality_summary,
    encode_target,
    load_raw_data,
    to_numeric_strict,
    validate_schema,
)


# --- Facts about the real file (verified in Phase 1) -------------------------


def test_real_file_shape_and_columns(raw_df):
    assert raw_df.shape == (7043, 21)
    assert list(raw_df.columns) == RAW_COLUMNS


def test_customer_ids_are_unique_and_not_features(raw_df):
    assert raw_df[ID_COLUMN].is_unique
    assert ID_COLUMN not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS


def test_baseline_churn_rate(raw_df):
    rate = encode_target(raw_df[TARGET_COLUMN]).mean()
    assert rate == pytest.approx(1869 / 7043)


def test_blank_total_charges_are_only_new_customers(raw_df):
    cleaned = clean_total_charges(raw_df)
    blanks = cleaned[cleaned["TotalCharges"].isna()]
    assert len(blanks) == 11
    assert (blanks["tenure"] == 0).all()
    assert (cleaned["tenure"] == 0).sum() == 11


def test_quality_summary_detects_hidden_blanks(raw_df):
    summary = data_quality_summary(raw_df).set_index("column")
    assert summary.loc["TotalCharges", "blank text"] == 11
    assert summary["missing (NaN)"].sum() == 0


# --- Loading -----------------------------------------------------------------


def test_load_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_raw_data(tmp_path / "nope.csv")


def test_load_rejects_file_with_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="Missing required column"):
        load_raw_data(path)


# --- Cleaning ----------------------------------------------------------------


def test_clean_total_charges_does_not_mutate_input(sample_df):
    before = sample_df["TotalCharges"].copy()
    cleaned = clean_total_charges(sample_df)
    pd.testing.assert_series_equal(sample_df["TotalCharges"], before)
    assert cleaned["TotalCharges"].dtype == float


def test_to_numeric_strict_blank_becomes_nan():
    result = to_numeric_strict(pd.Series(["1.5", " ", "", "20"]), "x")
    assert result.isna().tolist() == [False, True, True, False]
    assert result.iloc[0] == 1.5


def test_to_numeric_strict_rejects_garbage():
    with pytest.raises(SchemaError, match="non-numeric"):
        to_numeric_strict(pd.Series(["10", "12,5", "abc"]), "TotalCharges")


def test_to_numeric_strict_passes_numeric_through():
    result = to_numeric_strict(pd.Series([1, 2, np.nan]), "x")
    assert result.dtype == float


def test_encode_target_maps_yes_no():
    assert encode_target(pd.Series(["Yes", "No", " Yes "])).tolist() == [1, 0, 1]


def test_encode_target_rejects_unknown_labels():
    with pytest.raises(SchemaError):
        encode_target(pd.Series(["Yes", "Maybe"]))


# --- Validation --------------------------------------------------------------


def test_validate_accepts_features_without_target(sample_df):
    validate_schema(sample_df.drop(columns=[TARGET_COLUMN]), require_target=False)


def test_validate_requires_target_when_asked(sample_df):
    with pytest.raises(SchemaError, match="Churn"):
        validate_schema(sample_df.drop(columns=[TARGET_COLUMN]))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Contract", "Weekly", "unexpected value"),
        ("SeniorCitizen", 2, "unexpected value"),
        ("gender", " ", "blank"),
        ("tenure", -1, "negative"),
        ("tenure", 1.5, "whole number"),
        ("MonthlyCharges", "cheap", "non-numeric"),
        ("TotalCharges", "n/a", "non-numeric"),
        ("Churn", "Maybe", "Churn"),
    ],
)
def test_validate_rejects_invalid_values(sample_df, column, value, message):
    sample_df[column] = sample_df[column].astype(object)
    sample_df.loc[sample_df.index[0], column] = value
    with pytest.raises(SchemaError, match=message):
        validate_schema(sample_df)


def test_validate_reports_all_problems_at_once(sample_df):
    sample_df["Contract"] = "Weekly"
    sample_df["tenure"] = -5
    with pytest.raises(SchemaError) as exc:
        validate_schema(sample_df)
    assert "Contract" in str(exc.value) and "tenure" in str(exc.value)


def test_validate_rejects_empty_and_non_dataframe(sample_df):
    with pytest.raises(SchemaError, match="no rows"):
        validate_schema(sample_df.iloc[0:0])
    with pytest.raises(SchemaError, match="DataFrame"):
        validate_schema([1, 2, 3])


# --- Summaries ---------------------------------------------------------------


def test_churn_rate_by_matches_manual_calculation(raw_df):
    rates = churn_rate_by(raw_df, "Contract").set_index("Contract")
    manual = (raw_df[TARGET_COLUMN] == "Yes").groupby(raw_df["Contract"]).mean()
    pd.testing.assert_series_equal(
        rates["churn_rate"].sort_index(), manual.sort_index(), check_names=False
    )
    assert rates["customers"].sum() == len(raw_df)
    assert rates["churn_rate"].is_monotonic_decreasing
