import pandas as pd
import pytest

from src.config import RAW_DATA_PATH
from src.data_processing import load_raw_data


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The real dataset, loaded once per test session."""
    if not RAW_DATA_PATH.is_file():
        pytest.fail(f"Dataset missing at {RAW_DATA_PATH}")
    return load_raw_data()


@pytest.fixture
def sample_df(raw_df) -> pd.DataFrame:
    """A small, mutable copy of real rows for negative tests."""
    return raw_df.head(20).copy()
