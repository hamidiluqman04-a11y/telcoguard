"""Project-wide constants: paths, seed, and the expected dataset schema.

Everything that more than one notebook or module needs lives here so there is
a single source of truth (e.g. the random seed or the list of feature columns).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# The real file supplied with the project (verified in Phase 1).
RAW_DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"

PROCESSED_DIR = DATA_DIR / "processed"
# Which customerIDs belong to train vs test, written once by Notebook 2 and
# reused by every later notebook so all models see exactly the same split.
SPLIT_PATH = PROCESSED_DIR / "train_test_split.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# SeniorCitizen is stored as 0/1 in the raw file; every other categorical
# column is stored as text. Allowed levels were taken from the actual data.
CATEGORY_LEVELS: dict[str, set] = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}
CATEGORICAL_COLUMNS = list(CATEGORY_LEVELS)

# Add-on services that only exist for internet customers.
INTERNET_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Predictive features: everything except the ID and the target.
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

# Column order of the raw CSV.
RAW_COLUMNS = [
    ID_COLUMN,
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET_COLUMN,
]
