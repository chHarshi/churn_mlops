"""
Step 1: first look at the Credit Card Customers (BankChurners) data.
Run: python eda.py   (put BankChurners.csv in the same folder)
"""
import pandas as pd
CSV = "data/raw/BankChurners.csv"
df = pd.read_csv(CSV)
 
print("=== 1. Shape and types ===")
print(df.shape)
print(df.dtypes, "\n")
 
print("=== 2. Column names ===")
print(list(df.columns), "\n")
 
print("=== 3. First rows ===")
print(df.head(), "\n")
 
# --- The dataset creator explicitly says to drop the last two columns: they
# leak the target through a Naive Bayes classifier's own output. This is the
# built-in "did you actually read the dataset description" test — same idea
# as the leakage you caught in your receipt project. We also drop CLIENTNUM,
# which is just an ID with no predictive signal.
leak_cols = [c for c in df.columns if c.startswith("Naive_Bayes_Classifier")]
print("=== 4. Columns to drop (leakage / ID, not predictive) ===")
print("Naive Bayes leakage columns found:", leak_cols)
df = df.drop(columns=leak_cols + ["CLIENTNUM"])
print("Shape after dropping:", df.shape, "\n")
 
print("=== 5. Target: Attrition_Flag ===")
print(df["Attrition_Flag"].value_counts())
print(df["Attrition_Flag"].value_counts(normalize=True).round(3), "\n")
# Make a clean binary target: 1 = churned ("Attrited Customer"), 0 = retained
df["Churn"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
 
print("=== 6. Data quality ===")
print("Duplicate rows:", df.duplicated().sum())
print("Missing values per column:\n", df.isna().sum()[lambda s: s > 0])
print("Any 'Unknown' category values (hidden missingness)?")
for col in df.select_dtypes(include="object").columns:
    n_unknown = (df[col] == "Unknown").sum()
    if n_unknown:
        print(f"  {col}: {n_unknown} 'Unknown' values")
print()
 
print("=== 7. Churn rate by category ===")
for col in ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]:
    rate = df.groupby(col)["Churn"].mean().round(3).sort_values(ascending=False)
    print(f"\n{col}:\n{rate}")
 
print("\n=== 8. Numeric features, split by churn ===")
numeric_cols = [
    "Customer_Age", "Months_on_book", "Total_Relationship_Count",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit",
    "Total_Revolving_Bal", "Total_Trans_Amt", "Total_Trans_Ct",
    "Avg_Utilization_Ratio",
]
print(df.groupby("Churn")[numeric_cols].median())
 