# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_CUSTOMER = {
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


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input_returns_200():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "risk_tier" in body


def test_predict_probability_in_valid_range():
    response = client.post("/predict", json=VALID_CUSTOMER)
    probability = response.json()["churn_probability"]
    assert 0.0 <= probability <= 1.0


def test_predict_risk_tier_is_valid_category():
    response = client.post("/predict", json=VALID_CUSTOMER)
    tier = response.json()["risk_tier"]
    assert tier in {"low", "medium", "high"}


def test_predict_rejects_negative_age():
    bad_customer = {**VALID_CUSTOMER, "Customer_Age": -5}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422  # Pydantic validation error


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "Gender"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_utilization_ratio():
    bad_customer = {**VALID_CUSTOMER, "Avg_Utilization_Ratio": 1.5}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422