from fastapi.testclient import TestClient

from churn_mlops.api import app

client = TestClient(app)

VALID_CUSTOMER = {
    "Customer_Age": 45,
    "Gender": "M",
    "Dependent_count": 3,
    "Education_Level": "High School",
    "Marital_Status": "Married",
    "Income_Category": "$60K - $80K",
    "Card_Category": "Blue",
    "Months_on_book": 39,
    "Total_Relationship_Count": 5,
    "Months_Inactive_12_mon": 1,
    "Contacts_Count_12_mon": 3,
    "Credit_Limit": 12691,
    "Total_Revolving_Bal": 777,
    "Avg_Open_To_Buy": 11914,
    "Total_Amt_Chng_Q4_Q1": 1.335,
    "Total_Trans_Amt": 1144,
    "Total_Trans_Ct": 42,
    "Total_Ct_Chng_Q4_Q1": 1.625,
    "Avg_Utilization_Ratio": 0.061,
}


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_unknown_category():
    bad_customer = {**VALID_CUSTOMER, "Gender": "X"}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in VALID_CUSTOMER.items() if k != "Customer_Age"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422