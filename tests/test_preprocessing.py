import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS, TARGET_COLUMN
from src.data_processing import (
    SchemaError,
    apply_split,
    build_preprocessor,
    load_split,
    make_train_test_split,
    prepare_features,
    prepare_features_target,
    save_split,
)


@pytest.fixture(scope="module")
def xy(raw_df):
    return prepare_features_target(raw_df)


def test_features_exclude_id_and_target(xy):
    features, _ = xy
    assert list(features.columns) == FEATURE_COLUMNS
    assert ID_COLUMN not in features.columns and TARGET_COLUMN not in features.columns
    assert features.index.name == ID_COLUMN


def test_features_and_target_aligned(xy, raw_df):
    features, target = xy
    assert features.index.equals(target.index)
    assert target.sum() == (raw_df[TARGET_COLUMN] == "Yes").sum()


def test_prepare_features_strips_whitespace_in_categories(sample_df):
    sample_df["Contract"] = sample_df["Contract"].astype(object)
    sample_df.loc[sample_df.index[0], "Contract"] = "  Two year "
    assert prepare_features(sample_df)["Contract"].iloc[0] == "Two year"


def test_prepare_features_without_id_or_target(sample_df):
    features = prepare_features(sample_df.drop(columns=[ID_COLUMN, TARGET_COLUMN]))
    assert list(features.columns) == FEATURE_COLUMNS


def test_prepare_features_rejects_invalid_input(sample_df):
    sample_df["PaymentMethod"] = "Bitcoin"
    with pytest.raises(SchemaError):
        prepare_features(sample_df)


def test_split_is_stratified_and_reproducible(xy):
    features, target = xy
    X_tr, X_te, y_tr, y_te = make_train_test_split(features, target)
    X_tr2, _, _, _ = make_train_test_split(features, target)
    assert X_tr.index.equals(X_tr2.index)
    assert len(X_te) == pytest.approx(0.2 * len(features), abs=1)
    assert abs(y_tr.mean() - y_te.mean()) < 0.005
    assert X_tr.index.intersection(X_te.index).empty


def test_split_roundtrip(tmp_path, xy):
    features, target = xy
    X_tr, X_te, y_tr, y_te = make_train_test_split(features, target)
    path = save_split(X_tr.index, X_te.index, tmp_path / "split.csv")
    A_tr, A_te, b_tr, b_te = apply_split(features, target, load_split(path))
    assert A_tr.index.equals(X_tr.index) and A_te.index.equals(X_te.index)
    pd.testing.assert_series_equal(b_te, y_te)


def test_save_split_rejects_overlap(tmp_path):
    with pytest.raises(ValueError, match="both train and test"):
        save_split(["a", "b"], ["b", "c"], tmp_path / "split.csv")


def test_apply_split_rejects_mismatched_ids(tmp_path, xy):
    features, target = xy
    split = pd.DataFrame({ID_COLUMN: ["x-1", "x-2"], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="does not match"):
        apply_split(features, target, split)


def test_load_split_missing_file_has_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="02_data_preprocessing"):
        load_split(tmp_path / "missing.csv")


def test_saved_project_split_matches_data(xy):
    """The split written by Notebook 2 must cover exactly this dataset."""
    features, target = xy
    X_tr, X_te, _, _ = apply_split(features, target, load_split())
    assert len(X_tr) + len(X_te) == len(features)


def test_preprocessor_imputes_zero_and_learns_from_train_only(xy):
    features, target = xy
    X_tr, X_te, _, _ = make_train_test_split(features, target)
    pre = build_preprocessor().fit(X_tr)
    scaler = pre.named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, X_tr[NUMERIC_COLUMNS].fillna(0).mean())

    out = pre.transform(features)
    assert not np.isnan(out).any()
    # A blank TotalCharges is transformed exactly like a literal 0.
    blank_row = features[features["TotalCharges"].isna()].head(1)
    zero_row = blank_row.assign(TotalCharges=0.0)
    np.testing.assert_allclose(pre.transform(blank_row), pre.transform(zero_row))


def test_preprocessor_ignores_unknown_categories(xy):
    features, target = xy
    pre = build_preprocessor().fit(features)
    row = features.head(1).assign(PaymentMethod="Bitcoin")
    names = pre.get_feature_names_out()
    encoded = pd.Series(pre.transform(row)[0], index=names)
    assert encoded[[n for n in names if "PaymentMethod" in n]].sum() == 0
