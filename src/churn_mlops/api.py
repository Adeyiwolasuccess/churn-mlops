from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from churn_mlops.predict import load_artifacts, preprocess_new_customer, predict_churn

app = FastAPI(title="Churn Prediction API")

model, encoders, scaler = load_artifacts()


class CustomerFeatures(BaseModel):
    Customer_Age: float
    Dependent_count: float
    Months_on_book: float
    Total_Relationship_Count: float
    Months_Inactive_12_mon: float
    Contacts_Count_12_mon: float
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: float
    Total_Trans_Ct: float
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float
    Education_Level: Literal[
        "Uneducated", "High School", "College", "Graduate",
        "Post-Graduate", "Doctorate", "Unknown",
    ]
    Income_Category: Literal[
        "Less than $40K", "$40K - $60K", "$60K - $80K",
        "$80K - $120K", "$120K +", "Unknown",
    ]
    Gender: Literal["M", "F"]
    Marital_Status: Literal["Married", "Single", "Divorced", "Unknown"]
    Card_Category: Literal["Blue", "Silver", "Gold", "Platinum"]


@app.get("/")
def root():
    """
    Simple health check endpoint.
    """
    return {"status": "Churn prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    """
    Accepts raw customer data, runs it through the same preprocessing
    pipeline used at training time, and returns the predicted class
    and churn probability.
    """
    raw_data = customer.model_dump()
    processed = preprocess_new_customer(raw_data, encoders, scaler)
    predictions, probabilities = predict_churn(model, processed)

    return {
        "prediction": int(predictions[0]),
        "churn_probability": float(probabilities[0]),
    }