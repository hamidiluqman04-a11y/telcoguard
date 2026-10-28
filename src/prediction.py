"""Scoring customers with the final saved pipeline.

Every input goes through the same schema validation as the training data, so
a malformed file produces a clear error instead of silently wrong predictions.
"""

import io

import numpy as np
import pandas as pd

from src.config import ID_COLUMN, MODELS_DIR, TARGET_COLUMN
from src.data_processing import SchemaError, encode_target, prepare_features
from src.evaluation import load_final_selection
from src.training import load_model_artifact

MAX_UPLOAD_ROWS = 200_000
RISK_LABELS = ["Low", "Medium", "High"]


def load_final_model(models_dir=MODELS_DIR):
    """The pipeline chosen in Notebook 4, its selection record and metadata."""
    selection = load_final_selection(models_dir / "final_model.json")
    pipeline, metadata = load_model_artifact(selection["model_name"], models_dir)
    return pipeline, selection, metadata


def read_customer_csv(content: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV, turning low-level parser errors into readable ones."""
    if not content or not content.strip():
        raise SchemaError("The uploaded file is empty.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SchemaError("The file is not UTF-8 encoded text. Save it as UTF-8 CSV.") from exc
    try:
        df = pd.read_csv(io.StringIO(text), dtype={ID_COLUMN: "string"})
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise SchemaError(f"The file could not be read as CSV: {exc}") from exc
    if df.empty:
        raise SchemaError("The CSV has a header but no customer rows.")
    if len(df) > MAX_UPLOAD_ROWS:
        raise SchemaError(f"The CSV has {len(df):,} rows; the limit is {MAX_UPLOAD_ROWS:,}.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def score_customers(pipeline, raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Validate raw customer rows and return their features plus predictions.

    The result is indexed by customerID (or `row_<n>` if the file has no ID
    column) and contains every model feature, `churn_probability`, `flagged`
    (probability >= threshold) and, if the input has a Churn column,
    `actual_churn` (0/1) for checking.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")
    if ID_COLUMN in raw.columns:
        ids = raw[ID_COLUMN].astype("string").str.strip()
        if ids.isna().any() or (ids == "").any():
            raise SchemaError(f"'{ID_COLUMN}' has blank values.")
        duplicated = ids[ids.duplicated()].unique().tolist()
        if duplicated:
            raise SchemaError(f"Duplicate {ID_COLUMN} values, e.g. {duplicated[:5]}.")

    features = prepare_features(raw)
    if ID_COLUMN not in raw.columns:
        features.index = pd.Index([f"row_{i + 1}" for i in range(len(features))], name=ID_COLUMN)

    proba = pipeline.predict_proba(features)[:, 1]
    if np.isnan(proba).any():
        raise RuntimeError("The model returned NaN probabilities; check the input data.")

    scored = features.copy()
    scored.insert(0, "churn_probability", proba)
    scored.insert(1, "flagged", proba >= threshold)
    if TARGET_COLUMN in raw.columns:
        actual = encode_target(raw[TARGET_COLUMN])
        scored["actual_churn"] = actual.to_numpy()
    return scored


def assign_risk_category(proba, medium_from: float, high_from: float) -> pd.Series:
    """Low < medium_from <= Medium < high_from <= High."""
    if not 0 < medium_from < high_from < 1:
        raise ValueError("Cut-offs must satisfy 0 < medium_from < high_from < 1.")
    proba = proba if isinstance(proba, pd.Series) else pd.Series(proba)
    category = np.select(
        [proba >= high_from, proba >= medium_from], ["High", "Medium"], default="Low"
    )
    return pd.Series(
        pd.Categorical(category, categories=RISK_LABELS, ordered=True),
        index=proba.index,
        name="risk_category",
    )


def upload_template(raw_rows: pd.DataFrame) -> bytes:
    """A small example CSV (without the Churn column) users can copy."""
    return raw_rows.drop(columns=[TARGET_COLUMN], errors="ignore").to_csv(index=False).encode()
