# src/churn/features.py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"

NUMERIC_FEATURES = [
    "Customer_Age", "Dependent_count", "Months_on_book",
    "Total_Relationship_Count", "Months_Inactive_12_mon",
    "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
    "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
    "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
]

CATEGORICAL_FEATURES = [
    "Gender", "Education_Level", "Marital_Status",
    "Income_Category", "Card_Category",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    One ColumnTransformer that scales numeric columns and one-hot encodes
    categorical ones (including "Unknown" as its own category, not imputed
    away — see the note in data.py).
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """
    Wraps any sklearn-compatible model with the shared preprocessor. This
    whole Pipeline — not just the model — is what gets saved and loaded,
    so training and serving are guaranteed to preprocess identically.
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
# at the bottom of features.py, or in a throwaway script
if __name__ == "__main__":
    from src.churn.data import load_clean

    df = load_clean()
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    pre = build_preprocessor()
    X_transformed = pre.fit_transform(X)
    print("Shape before:", X.shape)
    print("Shape after:", X_transformed.shape)