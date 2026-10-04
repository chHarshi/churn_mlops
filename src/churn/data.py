# src/churn/data.py
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "BankChurners.csv"

TARGET_COL = "Churn"
DROP_COLS = ["CLIENTNUM", "Attrition_Flag"]  # ID + the source of the target


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    leak_cols = [c for c in df.columns if c.startswith("Naive_Bayes_Classifier")]
    df = df.drop(columns=leak_cols)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    df = df.drop(columns=DROP_COLS)
    # Leave "Unknown" as its own category for now rather than imputing —
    # it may itself carry signal (e.g. people who skip income disclosure).
    # Document this choice; it's a legitimate modeling decision either way.
    return df


def load_clean(path: Path = RAW_CSV) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    df = load_clean()
    print(df.shape)
    print(df[TARGET_COL].value_counts(normalize=True).round(3))