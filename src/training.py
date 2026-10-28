"""Model definitions, cross-validated tuning and artifact saving/loading.

Tuning only ever sees the training set: GridSearchCV splits it into folds and
refits the full preprocessing + model pipeline inside each fold.
"""

import json
import platform
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, FEATURE_COLUMNS, MODELS_DIR, RANDOM_STATE
from src.data_processing import build_preprocessor

# Metrics computed in every CV fold. Threshold-based ones (precision, recall,
# F1) use the default 0.5 cut-off; threshold choice is studied in Notebook 4.
CV_SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}
# Hyperparameters are chosen by PR-AUC (average precision). It focuses on how
# well churners are ranked and, unlike accuracy, is not inflated by the large
# number of non-churners.
REFIT_METRIC = "average_precision"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    estimator: object
    param_grid: dict = field(default_factory=dict)


def model_specs(random_state: int = RANDOM_STATE) -> dict[str, ModelSpec]:
    """The three candidate models and their (deliberately small) search grids.

    Grid keys are prefixed with `model__` because they address the estimator
    step inside the pipeline.
    """
    specs = [
        ModelSpec(
            "logistic_regression",
            "Logistic Regression",
            LogisticRegression(max_iter=5000, random_state=random_state),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1),
            {
                "model__max_depth": [6, 10, None],
                "model__min_samples_leaf": [1, 5, 10, 20],
                "model__class_weight": [None, "balanced_subsample"],
            },
        ),
        ModelSpec(
            "hist_gradient_boosting",
            "Histogram Gradient Boosting",
            HistGradientBoostingClassifier(
                max_iter=200, early_stopping=False, random_state=random_state
            ),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_leaf_nodes": [4, 8, 31],
                "model__min_samples_leaf": [20, 50, 100],
                "model__class_weight": [None, "balanced"],
            },
        ),
    ]
    return {spec.name: spec for spec in specs}


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing and model as one object, so it is fitted and saved together."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def make_cv(random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)


def _check_training_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError(f"X must have exactly the feature columns {FEATURE_COLUMNS}.")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must have the same rows in the same order.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y must be binary 0/1 and contain both classes.")


def tune_model(
    spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series, n_jobs: int = -1
) -> GridSearchCV:
    """Grid search with stratified K-fold CV on the training set only."""
    _check_training_inputs(X_train, y_train)
    search = GridSearchCV(
        build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        scoring=CV_SCORING,
        refit=REFIT_METRIC,
        cv=make_cv(),
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )
    start = time.perf_counter()
    search.fit(X_train, y_train)
    search.fit_seconds_ = time.perf_counter() - start
    return search


def baseline_cv_scores(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    """CV scores of a model that ignores the features (predicts the base rate)."""
    _check_training_inputs(X_train, y_train)
    with warnings.catch_warnings():
        # Precision is undefined when nothing is predicted positive; that is
        # exactly the point of this baseline, so report it as 0.
        warnings.simplefilter("ignore")
        result = cross_validate(
            build_pipeline(DummyClassifier(strategy="prior")),
            X_train,
            y_train,
            scoring=CV_SCORING,
            cv=make_cv(),
        )
    return {metric: float(result[f"test_{metric}"].mean()) for metric in CV_SCORING}


def best_cv_scores(search: GridSearchCV) -> dict[str, dict[str, float]]:
    """Mean and std of every CV metric for the selected hyperparameters."""
    results = search.cv_results_
    i = search.best_index_
    return {
        metric: {
            "mean": float(results[f"mean_test_{metric}"][i]),
            "std": float(results[f"std_test_{metric}"][i]),
            "train_mean": float(results[f"mean_train_{metric}"][i]),
        }
        for metric in CV_SCORING
    }


def fold_scores(search: GridSearchCV, metric: str = REFIT_METRIC) -> list[float]:
    """Per-fold validation scores of the selected hyperparameters."""
    i = search.best_index_
    return [
        float(search.cv_results_[f"split{k}_test_{metric}"][i])
        for k in range(search.n_splits_)
    ]


def params_at_grid_edge(search: GridSearchCV) -> list[str]:
    """Numeric hyperparameters whose chosen value is the smallest or largest
    tried. A boundary choice hints that a wider grid might do better.

    `None` in a numeric grid (e.g. max_depth=None) means "unlimited", so it
    counts as larger than every number and the largest number is not an edge.
    """
    flagged = []
    for name, chosen in search.best_params_.items():
        grid = search.param_grid[name]
        numeric = [v for v in grid if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numeric) < 2 or chosen not in numeric:
            continue
        upper_is_edge = None not in grid
        if chosen == min(numeric) or (upper_is_edge and chosen == max(numeric)):
            flagged.append(name)
    return flagged


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_model_artifact(
    search: GridSearchCV,
    spec: ModelSpec,
    extra_metadata: dict,
    models_dir: Path | str = MODELS_DIR,
) -> tuple[Path, Path]:
    """Save the refitted best pipeline (.joblib) and its metadata (.json)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{spec.name}.joblib"
    meta_path = models_dir / f"{spec.name}.json"

    metadata = {
        "model_name": spec.name,
        "model_label": spec.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "selection_metric": REFIT_METRIC,
        "cv_folds": search.n_splits_,
        "cv_scores": best_cv_scores(search),
        "cv_fold_scores": {m: fold_scores(search, m) for m in CV_SCORING},
        "n_candidates": len(search.cv_results_["params"]),
        "param_grid": {k: [_jsonable(v) for v in vs] for k, vs in search.param_grid.items()},
        "params_at_grid_edge": params_at_grid_edge(search),
        "fit_seconds": round(getattr(search, "fit_seconds_", float("nan")), 2),
        "feature_columns": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        **extra_metadata,
    }
    joblib.dump(search.best_estimator_, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, meta_path


def load_model_artifact(name: str, models_dir: Path | str = MODELS_DIR):
    """Load a saved pipeline and its metadata, warning on version mismatch."""
    models_dir = Path(models_dir)
    model_path = models_dir / f"{name}.joblib"
    meta_path = models_dir / f"{name}.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Model artifact '{name}' not found in {models_dir}. "
            "Run notebooks/03_model_training.py first."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    saved_version = metadata.get("versions", {}).get("scikit-learn")
    if saved_version != sklearn.__version__:
        warnings.warn(
            f"Model '{name}' was saved with scikit-learn {saved_version}, "
            f"but {sklearn.__version__} is installed. Predictions may differ; retrain.",
            stacklevel=2,
        )
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"{model_path} does not contain a scikit-learn Pipeline.")
    return pipeline, metadata
