# src/churn/predict.py
from pathlib import Path
import joblib
import pandas as pd

from src.churn.features import ALL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

HIGH_RISK_THRESHOLD = 0.6
MEDIUM_RISK_THRESHOLD = 0.3

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def risk_tier(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "high"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def predict_one(customer: dict) -> dict:
    model = get_model()
    X = pd.DataFrame([customer])[ALL_FEATURES]
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "churn_probability": round(probability, 4),
        "risk_tier": risk_tier(probability),
    }
if __name__ == "__main__":
    sample = {
        "Customer_Age": 45, "Dependent_count": 3, "Months_on_book": 36,
        "Total_Relationship_Count": 3, "Months_Inactive_12_mon": 3,
        "Contacts_Count_12_mon": 4, "Credit_Limit": 5000.0,
        "Total_Revolving_Bal": 500, "Avg_Open_To_Buy": 4500.0,
        "Total_Amt_Chng_Q4_Q1": 0.6, "Total_Trans_Amt": 2500,
        "Total_Trans_Ct": 40, "Total_Ct_Chng_Q4_Q1": 0.5,
        "Avg_Utilization_Ratio": 0.1, "Gender": "M",
        "Education_Level": "Graduate", "Marital_Status": "Married",
        "Income_Category": "$40K - $60K", "Card_Category": "Blue",
    }
    print(predict_one(sample))