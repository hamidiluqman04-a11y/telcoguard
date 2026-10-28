"""Metrics, uncertainty estimates and decision-threshold selection.

Rule followed throughout: thresholds are *chosen* on out-of-fold predictions
from the training set and only *reported* on the test set, so the test set
never influences a decision.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from src.config import MODELS_DIR, RANDOM_STATE
from src.training import make_cv

DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
FINAL_SELECTION_PATH = MODELS_DIR / "final_model.json"


def _check(y_true, proba):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if y_true.shape != proba.shape:
        raise ValueError("y_true and proba must have the same length.")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError("y_true must be 0/1.")
    if np.isnan(proba).any() or (proba < 0).any() or (proba > 1).any():
        raise ValueError("Probabilities must be in [0, 1] with no NaN.")
    return y_true, proba


def confusion_counts(y_true, proba, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """TN/FP/FN/TP when customers with proba >= threshold are flagged."""
    y_true, proba = _check(y_true, proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def classification_metrics(y_true, proba, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Threshold-free and threshold-based metrics in one dictionary."""
    y_true, proba = _check(y_true, proba)
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "flagged_share": float(pred.mean()),
        **confusion_counts(y_true, proba, threshold),
    }


def bootstrap_ci(
    y_true,
    proba,
    metric=average_precision_score,
    n_resamples: int = 1000,
    level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Percentile bootstrap interval: how much the score could move with a
    different sample of test customers of the same size."""
    y_true, proba = _check(y_true, proba)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # a resample with one class has no defined AUC
        scores.append(metric(y_true[idx], proba[idx]))
    alpha = (1 - level) / 2
    low, high = np.quantile(scores, [alpha, 1 - alpha])
    return float(low), float(high)


def out_of_fold_proba(pipeline, X_train, y_train) -> pd.Series:
    """Churn probabilities for every training customer from a model that did
    not see that customer (5-fold CV with the pipeline's tuned settings).

    The hyperparameters were selected with the same folds, so these scores are
    very slightly optimistic, but no test data is involved.
    """
    proba = cross_val_predict(
        clone(pipeline), X_train, y_train, cv=make_cv(), method="predict_proba"
    )[:, 1]
    return pd.Series(proba, index=X_train.index, name="oof_proba")


def threshold_table(y_true, proba, thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    """Precision, recall, F1 and share flagged at each candidate threshold."""
    y_true, proba = _check(y_true, proba)
    rows = []
    for t in thresholds:
        pred = proba >= t
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "flagged_share": float(pred.mean()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return pd.DataFrame(rows)


def threshold_max_f1(y_true, proba, thresholds=THRESHOLD_GRID) -> float:
    table = threshold_table(y_true, proba, thresholds)
    return float(table.loc[table["f1"].idxmax(), "threshold"])


def threshold_for_recall(y_true, proba, target_recall: float, thresholds=THRESHOLD_GRID) -> float:
    """Highest threshold (i.e. fewest customers flagged) that still reaches
    the target recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1].")
    table = threshold_table(y_true, proba, thresholds)
    ok = table[table["recall"] >= target_recall]
    if ok.empty:
        return float(table["threshold"].min())
    return float(ok["threshold"].max())


def expected_cost(
    y_true,
    proba,
    threshold: float,
    contact_cost: float,
    churn_loss: float,
    save_rate: float,
) -> float:
    """Total cost of a retention campaign under *assumed* economics.

    - Every flagged customer costs `contact_cost` (offer, call, discount).
    - A churner who is flagged is saved with probability `save_rate`; if not
      saved we still lose `churn_loss`.
    - A churner who is not flagged is lost: `churn_loss`.
    """
    if contact_cost < 0 or churn_loss < 0 or not 0 <= save_rate <= 1:
        raise ValueError("Costs must be >= 0 and save_rate in [0, 1].")
    c = confusion_counts(y_true, proba, threshold)
    flagged = c["tp"] + c["fp"]
    return float(
        flagged * contact_cost
        + c["tp"] * (1 - save_rate) * churn_loss
        + c["fn"] * churn_loss
    )


def threshold_min_cost(
    y_true, proba, contact_cost, churn_loss, save_rate, thresholds=THRESHOLD_GRID
) -> float:
    costs = [
        expected_cost(y_true, proba, t, contact_cost, churn_loss, save_rate) for t in thresholds
    ]
    return float(thresholds[int(np.argmin(costs))])


def save_final_selection(selection: dict, path: Path | str = FINAL_SELECTION_PATH) -> Path:
    """Record which model and threshold the insights notebook should use."""
    required = {"model_name", "threshold", "threshold_policy"}
    missing = required - set(selection)
    if missing:
        raise ValueError(f"Selection is missing {sorted(missing)}.")
    if not 0 < selection["threshold"] < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return path


def load_final_selection(path: Path | str = FINAL_SELECTION_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No final model selection at {path}. Run notebooks/04_model_evaluation.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
