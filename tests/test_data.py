# tests/test_data.py
import pandas as pd
from src.churn.data import clean, TARGET_COL, DROP_COLS


def _make_fake_raw_df():
    return pd.DataFrame({
        "CLIENTNUM": [1, 2],
        "Attrition_Flag": ["Existing Customer", "Attrited Customer"],
        "Customer_Age": [40, 55],
        "Gender": ["M", "F"],
        "Education_Level": ["Graduate", "Unknown"],
        "Marital_Status": ["Married", "Single"],
        "Income_Category": ["$40K - $60K", "Unknown"],
        "Card_Category": ["Blue", "Gold"],
    })


def test_clean_creates_binary_churn_column():
    df = clean(_make_fake_raw_df())
    assert TARGET_COL in df.columns
    assert set(df[TARGET_COL].unique()) <= {0, 1}
    assert df[TARGET_COL].tolist() == [0, 1]  # Existing=0, Attrited=1


def test_clean_drops_leakage_and_id_columns():
    df = clean(_make_fake_raw_df())
    for col in DROP_COLS:
        assert col not in df.columns


def test_clean_preserves_row_count():
    raw = _make_fake_raw_df()
    df = clean(raw)
    assert len(df) == len(raw)