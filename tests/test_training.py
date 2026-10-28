import json
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, ID_COLUMN, MODELS_DIR, SPLIT_PATH
from src.data_processing import (
    apply_split,
    file_sha256,
    load_split,
    prepare_features_target,
)
from src.training import (
    ModelSpec,
    baseline_cv_scores,
    best_cv_scores,
    build_pipeline,
    fold_scores,
    load_model_artifact,
    model_specs,
    params_at_grid_edge,
    save_model_artifact,
    tune_model,
)


@pytest.fixture(scope="module")
def train_xy(raw_df):
    features, target = prepare_features_target(raw_df)
    X_tr, _, y_tr, _ = apply_split(features, target, load_split())
    return X_tr, y_tr


@pytest.fixture(scope="module")
def small_search(train_xy):
    """A quick, real grid search on a subsample (keeps the suite fast)."""
    X_tr, y_tr = train_xy
    X_small, y_small = X_tr.iloc[:1500], y_tr.iloc[:1500]
    spec = ModelSpec(
        "tiny_lr",
        "Tiny LR",
        LogisticRegression(max_iter=2000),
        {"model__C": [0.1, 1.0]},
    )
    return spec, tune_model(spec, X_small, y_small, n_jobs=1)


def test_three_model_families_are_defined():
    specs = model_specs()
    assert set(specs) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    for spec in specs.values():
        assert all(k.startswith("model__") for k in spec.param_grid)


def test_random_seeds_are_fixed():
    for spec in model_specs().values():
        assert spec.estimator.get_params()["random_state"] is not None


def test_pipeline_contains_preprocessing():
    pipe = build_pipeline(LogisticRegression())
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["preprocess", "model"]


def test_tune_rejects_bad_inputs(train_xy):
    X_tr, y_tr = train_xy
    spec = model_specs()["logistic_regression"]
    with pytest.raises(ValueError, match="feature columns"):
        tune_model(spec, X_tr.drop(columns=["tenure"]), y_tr)
    with pytest.raises(ValueError, match="same rows"):
        tune_model(spec, X_tr, y_tr.iloc[::-1])
    with pytest.raises(ValueError, match="binary"):
        tune_model(spec, X_tr, y_tr * 0)


def test_baseline_has_no_skill(train_xy):
    X_tr, y_tr = train_xy
    scores = baseline_cv_scores(X_tr, y_tr)
    assert scores["roc_auc"] == pytest.approx(0.5)
    assert scores["recall"] == 0.0
    assert scores["average_precision"] == pytest.approx(y_tr.mean(), abs=0.01)


def test_tuned_model_beats_baseline(small_search):
    _, search = small_search
    scores = best_cv_scores(search)
    assert scores["roc_auc"]["mean"] > 0.75
    assert len(fold_scores(search)) == 5


def test_pipeline_never_sees_customer_id(small_search):
    _, search = small_search
    seen = list(search.best_estimator_.feature_names_in_)
    assert seen == FEATURE_COLUMNS and ID_COLUMN not in seen


def test_artifact_roundtrip(tmp_path, small_search, train_xy):
    spec, search = small_search
    model_path, meta_path = save_model_artifact(search, spec, {"note": "test"}, tmp_path)
    pipe, meta = load_model_artifact(spec.name, tmp_path)
    X_tr, _ = train_xy
    np.testing.assert_allclose(
        pipe.predict_proba(X_tr.head(50)), search.best_estimator_.predict_proba(X_tr.head(50))
    )
    assert meta["note"] == "test"
    assert meta["feature_columns"] == FEATURE_COLUMNS
    assert meta["best_params"] == search.best_params_
    json.loads(meta_path.read_text())  # valid JSON


def test_load_missing_artifact_has_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="03_model_training"):
        load_model_artifact("nope", tmp_path)


def test_load_warns_on_version_mismatch(tmp_path, small_search):
    spec, search = small_search
    _, meta_path = save_model_artifact(search, spec, {}, tmp_path)
    meta = json.loads(meta_path.read_text())
    meta["versions"]["scikit-learn"] = "0.0.1"
    meta_path.write_text(json.dumps(meta))
    with pytest.warns(UserWarning, match="scikit-learn"):
        load_model_artifact(spec.name, tmp_path)


def test_params_at_grid_edge(small_search):
    _, search = small_search
    # With only two values every choice is on an edge.
    assert params_at_grid_edge(search) == ["model__C"]


@pytest.mark.parametrize(
    ("chosen", "flagged"),
    [(6, True), (10, False), (None, False)],
)
def test_params_at_grid_edge_treats_none_as_unlimited(chosen, flagged):
    fake = SimpleNamespace(
        best_params_={"model__max_depth": chosen},
        param_grid={"model__max_depth": [6, 10, None]},
    )
    assert (params_at_grid_edge(fake) == ["model__max_depth"]) is flagged


@pytest.mark.parametrize("name", list(model_specs()))
def test_saved_project_models_match_current_data(name):
    """Artifacts from Notebook 3 were trained on this exact data and split."""
    if not (MODELS_DIR / f"{name}.joblib").is_file():
        pytest.skip("Run notebooks/03_model_training.py to create model artifacts.")
    pipe, meta = load_model_artifact(name)
    assert meta["split_sha256"] == file_sha256(SPLIT_PATH)
    assert list(pipe.feature_names_in_) == FEATURE_COLUMNS
    assert meta["trained_on"].startswith("training split only")
