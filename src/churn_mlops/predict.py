import pandas as pd
import joblib

def load_model(model_path: str = "models/rf_model.joblib"):
    """
    Loads the trained Random Forest model from disk.
    """
    model = joblib.load(model_path)
    return model


def predict_churn(model, customer_data: pd.DataFrame):
    """
    Given a fitted model and a DataFrame of one or more customers
    (already fully processed: engineered, encoded, and scaled),
    returns the predicted class (0/1) and the predicted probability
    of churn for each row.
    """
    predictions = model.predict(customer_data)
    probabilities = model.predict_proba(customer_data)[:, 1]
    return predictions, probabilities