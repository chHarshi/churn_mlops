# api/schemas.py
from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    Customer_Age: int = Field(ge=18, le=100)
    Dependent_count: int = Field(ge=0, le=10)
    Months_on_book: int = Field(ge=0, le=60)
    Total_Relationship_Count: int = Field(ge=0, le=10)
    Months_Inactive_12_mon: int = Field(ge=0, le=12)
    Contacts_Count_12_mon: int = Field(ge=0, le=20)
    Credit_Limit: float = Field(ge=0)
    Total_Revolving_Bal: float = Field(ge=0)
    Avg_Open_To_Buy: float = Field(ge=0)
    Total_Amt_Chng_Q4_Q1: float = Field(ge=0)
    Total_Trans_Amt: float = Field(ge=0)
    Total_Trans_Ct: int = Field(ge=0)
    Total_Ct_Chng_Q4_Q1: float = Field(ge=0)
    Avg_Utilization_Ratio: float = Field(ge=0, le=1)
    Gender: str
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_tier: str