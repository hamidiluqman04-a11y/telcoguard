"""Loading, validating and cleaning the Telco churn data.

These functions are shared by the notebooks, the training code and the tests,
so the same rules are applied everywhere.
"""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NEGATIVE_LABEL,
    NUMERIC_COLUMNS,
    POSITIVE_LABEL,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SPLIT_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)


class SchemaError(ValueError):
    """Raised when a dataframe does not match the expected Telco schema."""


def _as_text(series: pd.Series) -> pd.Series:
    """Return a stripped string view of a column (missing stays missing)."""
    return series.astype("string").str.strip()


def blank_mask(series: pd.Series) -> pd.Series:
    """True where a value is missing or only whitespace (e.g. ' ')."""
    text = _as_text(series)
    return text.isna() | (text == "")


def to_numeric_strict(series: pd.Series, column: str) -> pd.Series:
    """Convert to float; blanks become NaN, anything else non-numeric raises.

    `pd.to_numeric(errors="coerce")` alone would silently turn typos such as
    "12,5" into NaN. Here only genuinely blank cells may become NaN.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    blanks = blank_mask(series)
    converted = pd.to_numeric(_as_text(series).where(~blanks), errors="coerce")
    bad = converted.isna() & ~blanks
    if bad.any():
        examples = series[bad].astype(str).unique()[:5].tolist()
        raise SchemaError(
            f"Column '{column}' has {int(bad.sum())} non-numeric value(s), "
            f"e.g. {examples}."
        )
    return converted.astype(float)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Check columns, category levels and numeric ranges.

    All problems are collected and raised together in one SchemaError so the
    user can fix a file in one go. Blank TotalCharges are allowed here; how they
    are handled is a preprocessing decision, not a schema violation.
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise SchemaError("The dataframe has no rows.")

    required = list(FEATURE_COLUMNS) + ([TARGET_COLUMN] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {missing}.")

    problems: list[str] = []

    for column, allowed in CATEGORY_LEVELS.items():
        values = df[column]
        if column == "SeniorCitizen":
            values = pd.to_numeric(values, errors="coerce")
            if values.isna().any():
                problems.append("SeniorCitizen must be 0 or 1 with no blanks.")
                continue
            values = values.astype(int)
        else:
            if blank_mask(values).any():
                problems.append(f"'{column}' has blank or missing values.")
                continue
            values = _as_text(values)
        unexpected = set(values.unique()) - allowed
        if unexpected:
            problems.append(
                f"'{column}' has unexpected value(s) {sorted(map(str, unexpected))}; "
                f"allowed: {sorted(map(str, allowed))}."
            )

    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        try:
            numeric = to_numeric_strict(df[column], column)
        except SchemaError as exc:
            problems.append(str(exc))
            continue
        if column != "TotalCharges" and numeric.isna().any():
            problems.append(f"'{column}' has missing values.")
        if (numeric < 0).any():
            problems.append(f"'{column}' has negative values.")
        if column == "tenure" and not (numeric.dropna() % 1 == 0).all():
            problems.append("'tenure' must be a whole number of months.")

    if require_target:
        target = _as_text(df[TARGET_COLUMN])
        unexpected = set(target.dropna().unique()) - {POSITIVE_LABEL, NEGATIVE_LABEL}
        if target.isna().any() or unexpected:
            problems.append(
                f"'{TARGET_COLUMN}' must contain only "
                f"'{POSITIVE_LABEL}'/'{NEGATIVE_LABEL}'."
            )

    if problems:
        raise SchemaError("Schema validation failed:\n- " + "\n- ".join(problems))


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as stored and validate its schema.

    The data is returned unmodified (TotalCharges is still text) so the
    exploration notebook can show the raw problems before they are cleaned.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Telco CSV in the data/ folder."
        )
    df = pd.read_csv(path, dtype={ID_COLUMN: "string"})
    validate_schema(df, require_target=True)
    return df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where TotalCharges is numeric (blank -> NaN).

    Filling the NaN values is left to the model pipeline so the rule is learned
    from training data only and is saved together with the model.
    """
    cleaned = df.copy()
    cleaned["TotalCharges"] = to_numeric_strict(df["TotalCharges"], "TotalCharges")
    return cleaned


def encode_target(series: pd.Series) -> pd.Series:
    """Map 'Yes'/'No' to 1/0, refusing any other value."""
    text = _as_text(series)
    mapping = {POSITIVE_LABEL: 1, NEGATIVE_LABEL: 0}
    encoded = text.map(mapping)
    if encoded.isna().any():
        raise SchemaError(
            f"Target contains values other than {list(mapping)}: "
            f"{text[encoded.isna()].unique()[:5].tolist()}"
        )
    return encoded.astype(int).rename(series.name)


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing, blank-text and unique counts."""
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing (NaN)": int(series.isna().sum()),
                "blank text": int((blank_mask(series) & series.notna()).sum()),
                "unique values": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customers, churners and churn rate for each level of `column`."""
    churned = encode_target(df[TARGET_COLUMN])
    grouped = (
        pd.DataFrame({column: df[column], "churned": churned})
        .groupby(column, observed=True)["churned"]
        .agg(customers="size", churned="sum")
        .reset_index()
    )
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    grouped["share_of_customers"] = grouped["customers"] / len(df)
    return grouped.sort_values("churn_rate", ascending=False, ignore_index=True)


# --- Features, target and split ----------------------------------------------


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw rows into the model's feature table.

    - Only FEATURE_COLUMNS are kept, so `customerID` and `Churn` can never
      reach the model.
    - The index is the customerID (when present) so predictions can be traced
      back to customers without the ID being a feature.
    - Text categories are stripped so " Yes" and "Yes" are the same level.
    - TotalCharges blanks stay NaN; the pipeline's imputer fills them.
    """
    validate_schema(df, require_target=False)
    cleaned = clean_total_charges(df)
    features = cleaned[FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        if column == "SeniorCitizen":
            features[column] = pd.to_numeric(features[column]).astype(int)
        else:
            features[column] = _as_text(features[column]).astype(str)
    features["tenure"] = to_numeric_strict(features["tenure"], "tenure")
    features["MonthlyCharges"] = to_numeric_strict(
        features["MonthlyCharges"], "MonthlyCharges"
    )
    if ID_COLUMN in df.columns:
        features.index = pd.Index(_as_text(df[ID_COLUMN]).astype(str), name=ID_COLUMN)
    return features


def prepare_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature table plus the 0/1 target, both indexed by customerID."""
    validate_schema(df, require_target=True)
    features = prepare_features(df)
    target = encode_target(df[TARGET_COLUMN])
    target.index = features.index
    return features, target


def make_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified split: train and test keep the same churn rate."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )


def save_split(train_ids, test_ids, path: Path | str = SPLIT_PATH) -> Path:
    """Write which customerIDs are train vs test to a CSV."""
    path = Path(path)
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"{len(overlap)} customer(s) appear in both train and test.")
    path.parent.mkdir(parents=True, exist_ok=True)
    split = pd.concat(
        [
            pd.DataFrame({ID_COLUMN: list(train_ids), "split": "train"}),
            pd.DataFrame({ID_COLUMN: list(test_ids), "split": "test"}),
        ],
        ignore_index=True,
    )
    split.to_csv(path, index=False)
    return path


def load_split(path: Path | str = SPLIT_PATH) -> pd.DataFrame:
    """Read the saved split; fail clearly if Notebook 2 has not been run."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found at {path}. Run notebooks/02_data_preprocessing.py first."
        )
    split = pd.read_csv(path, dtype={ID_COLUMN: str, "split": str})
    if set(split["split"]) - {"train", "test"} or split[ID_COLUMN].duplicated().any():
        raise ValueError(f"Split file {path} is malformed.")
    return split


def apply_split(features: pd.DataFrame, target: pd.Series, split: pd.DataFrame):
    """Return X_train, X_test, y_train, y_test using a saved split.

    The split must cover exactly the customers in `features`, otherwise the
    data or the split file has changed and results would not be comparable.
    """
    split_ids = set(split[ID_COLUMN])
    data_ids = set(features.index)
    if split_ids != data_ids:
        raise ValueError(
            "Saved split does not match the dataset "
            f"({len(split_ids - data_ids)} unknown IDs, "
            f"{len(data_ids - split_ids)} unassigned IDs). Re-run Notebook 2."
        )
    train_ids = split.loc[split["split"] == "train", ID_COLUMN]
    test_ids = split.loc[split["split"] == "test", ID_COLUMN]
    return (
        features.loc[train_ids],
        features.loc[test_ids],
        target.loc[train_ids],
        target.loc[test_ids],
    )


def file_sha256(path: Path | str) -> str:
    """Fingerprint of a file, stored in model metadata for reproducibility."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- Preprocessing -----------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """Unfitted ColumnTransformer for numeric and categorical features.

    Numeric: fill missing with 0 (the only missing values are TotalCharges for
    tenure-0 customers, who have genuinely been charged nothing) then
    standardise. Categorical: one-hot encode; categories never seen in
    training are encoded as all zeros instead of crashing.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
