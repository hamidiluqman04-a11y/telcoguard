import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.config import SPLIT_PATH
from src.data_processing import file_sha256
from src.evaluation import (
    FINAL_SELECTION_PATH,
    bootstrap_ci,
    classification_metrics,
    confusion_counts,
    expected_cost,
    load_final_selection,
    save_final_selection,
    threshold_for_recall,
    threshold_max_f1,
    threshold_min_cost,
    threshold_table,
)


@pytest.fixture
def toy():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.05, 0.9])
    return y, p


def test_confusion_counts(toy):
    y, p = toy
    assert confusion_counts(y, p, 0.5) == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert confusion_counts(y, p, 0.0)["tp"] == 3  # everyone flagged


def test_metrics_match_sklearn(toy):
    y, p = toy
    m = classification_metrics(y, p, 0.5)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["average_precision"] == pytest.approx(average_precision_score(y, p))
    assert m["f1"] == pytest.approx(f1_score(y, p >= 0.5))
    assert m["flagged_share"] == pytest.approx(3 / 8)


@pytest.mark.parametrize(
    "bad",
    [
        (np.array([0, 1]), np.array([0.2])),  # length mismatch
        (np.array([0, 2]), np.array([0.2, 0.3])),  # non-binary labels
        (np.array([0, 1]), np.array([0.2, 1.3])),  # probability > 1
        (np.array([0, 1]), np.array([0.2, np.nan])),  # NaN
    ],
)
def test_metrics_validate_inputs(bad):
    with pytest.raises(ValueError):
        classification_metrics(*bad)


def test_threshold_table_consistent_with_metrics(toy):
    y, p = toy
    row = threshold_table(y, p, [0.5]).iloc[0]
    m = classification_metrics(y, p, 0.5)
    assert row["precision"] == pytest.approx(m["precision"])
    assert row["recall"] == pytest.approx(m["recall"])


def test_threshold_max_f1_is_optimal(toy):
    y, p = toy
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    best = threshold_max_f1(y, p, grid)
    assert f1_score(y, p >= best) == pytest.approx(max(f1_score(y, p >= t) for t in grid))


def test_threshold_for_recall(toy):
    y, p = toy
    t = threshold_for_recall(y, p, 1.0)
    assert classification_metrics(y, p, t)["recall"] == 1.0
    # Highest such threshold: one step higher loses a churner.
    assert classification_metrics(y, p, t + 0.01)["recall"] < 1.0
    with pytest.raises(ValueError):
        threshold_for_recall(y, p, 0)


def test_expected_cost_hand_calculation(toy):
    y, p = toy  # at 0.5: tp=2, fp=1, fn=1
    cost = expected_cost(y, p, 0.5, contact_cost=10, churn_loss=100, save_rate=0.5)
    assert cost == pytest.approx(3 * 10 + 2 * 0.5 * 100 + 1 * 100)


def test_min_cost_threshold_extremes(toy):
    y, p = toy
    # Free contacts that always work -> every churner gets flagged.
    t_all = threshold_min_cost(y, p, 0, 100, 1.0)
    assert confusion_counts(y, p, t_all)["fn"] == 0
    # Contacts that never work -> nobody gets flagged.
    t_none = threshold_min_cost(y, p, 10, 100, 0.0)
    counts = confusion_counts(y, p, t_none)
    assert counts["tp"] + counts["fp"] == 0


def test_expected_cost_rejects_bad_assumptions(toy):
    with pytest.raises(ValueError):
        expected_cost(*toy, 0.5, contact_cost=-1, churn_loss=1, save_rate=0.5)
    with pytest.raises(ValueError):
        expected_cost(*toy, 0.5, contact_cost=1, churn_loss=1, save_rate=1.5)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    low, high = bootstrap_ci(y, p, roc_auc_score, n_resamples=300)
    assert low < roc_auc_score(y, p) < high
    assert bootstrap_ci(y, p, roc_auc_score, n_resamples=300) == (low, high)  # seeded


def test_final_selection_roundtrip_and_validation(tmp_path):
    path = save_final_selection(
        {"model_name": "m", "threshold": 0.3, "threshold_policy": "x"}, tmp_path / "f.json"
    )
    assert load_final_selection(path)["threshold"] == 0.3
    with pytest.raises(ValueError, match="missing"):
        save_final_selection({"model_name": "m"}, tmp_path / "g.json")
    with pytest.raises(ValueError, match="between 0 and 1"):
        save_final_selection(
            {"model_name": "m", "threshold": 1.0, "threshold_policy": "x"}, tmp_path / "h.json"
        )
    with pytest.raises(FileNotFoundError, match="04_model_evaluation"):
        load_final_selection(tmp_path / "missing.json")


def test_project_final_selection_is_consistent():
    """The selection written by Notebook 4 refers to the current split and
    was chosen without the test set (threshold from OOF, model from CV)."""
    if not FINAL_SELECTION_PATH.is_file():
        pytest.skip("Run notebooks/04_model_evaluation.py first.")
    sel = load_final_selection()
    assert sel["split_sha256"] == file_sha256(SPLIT_PATH)
    assert "out-of-fold" in sel["threshold_rule"]
    assert "cross-validated" in sel["selection_rule"]
    assert 0 < sel["threshold"] < 1
