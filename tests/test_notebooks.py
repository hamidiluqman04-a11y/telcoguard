"""Smoke tests: every notebook must run top to bottom without errors.

Notebooks run in numeric order (02 writes the split that 03 reads). Running
them rewrites data/processed/ and models/ with identical, reproducible content.
Skip these with `pytest -m "not slow"` for a quick run.
"""

import functools
import importlib.util

import pytest

from src.config import PROJECT_ROOT

NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("[0-9][0-9]_*.py"))

pytestmark = pytest.mark.slow


@functools.cache
def run_notebook(filename: str) -> dict:
    path = PROJECT_ROOT / "notebooks" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, defs = module.app.run()
    return defs


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path):
    assert run_notebook(path.name), f"{path.name} defined no variables"


def test_exploration_notebook_results():
    defs = run_notebook("01_data_exploration.py")
    assert defs["baseline_churn_rate"] == pytest.approx(1869 / 7043)
    assert len(defs["blank_total_rows"]) == 11
    assert defs["clean_df"]["TotalCharges"].dtype == float


def test_preprocessing_notebook_results():
    defs = run_notebook("02_data_preprocessing.py")
    assert len(defs["X_train"]) + len(defs["X_test"]) == 7043
    assert defs["X_train"].index.intersection(defs["X_test"].index).empty
    assert "customerID" not in defs["X_train"].columns


def test_training_notebook_results():
    defs = run_notebook("03_model_training.py")
    assert set(defs["searches"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert (defs["reload_check"]["reload reproduces predictions"] == "yes").all()
    # Training must never see test customers.
    split = defs["split"]
    test_ids = set(split.loc[split["split"] == "test", "customerID"])
    assert test_ids.isdisjoint(defs["X_train"].index)


def test_evaluation_notebook_results():
    run_notebook("03_model_training.py")  # 04 reads the models 03 writes
    defs = run_notebook("04_model_evaluation.py")
    sel = defs["final_selection"]
    assert sel["model_name"] == defs["model_names"][0]  # pre-registered CV winner
    assert len(defs["test_proba"][sel["model_name"]]) == sel["n_test"]
    # Threshold comes from training out-of-fold predictions, not the test set.
    from src.evaluation import threshold_max_f1

    assert sel["threshold"] == threshold_max_f1(
        defs["y_train"], defs["oof_proba"][sel["model_name"]]
    )


def test_insights_notebook_results():
    run_notebook("04_model_evaluation.py")  # 05 reads the final model 04 selects
    defs = run_notebook("05_churn_insights.py")
    scored = defs["risk_df"]
    # Default view scores the held-out test customers only.
    assert set(scored.index) == set(defs["X_test"].index)
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_category"].unique()) <= {"Low", "Medium", "High"}
    # The default filter keeps Medium + High and the explained customer is the top one.
    assert set(defs["filtered_df"]["risk_category"]) <= {"Medium", "High"}
    assert defs["selected_id"] == defs["filtered_df"].index[0]
