import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from churn_mlops.predict import load_model, predict_churn

app = FastAPI(title="Churn Prediction API")

model = load_model()


@app.get("/")
def root():
    """
    Simple health check endpoint.
    """
    return {"status": "Churn prediction API is running"}


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
    Avg_Trans_Size: float
    Education_Level: float
    Income_Category: float
    Age_Group: float
    Relationship_Contact_Ratio: float
    Gender_M: bool
    Marital_Status_Married: bool
    Marital_Status_Single: bool
    Marital_Status_Unknown: bool
    Card_Category_Gold: bool
    Card_Category_Platinum: bool
    Card_Category_Silver: bool


@app.post("/predict")
def predict(customer: CustomerFeatures):
    """
    Accepts one customer's already-processed features and returns
    the predicted class and churn probability.
    """
    data = pd.DataFrame([customer.model_dump()])
    data = data[model.feature_names_in_]  # reorder columns to match training order
    predictions, probabilities = predict_churn(model, data)

    return {
        "prediction": int(predictions[0]),
        "churn_probability": float(probabilities[0])
    }



    