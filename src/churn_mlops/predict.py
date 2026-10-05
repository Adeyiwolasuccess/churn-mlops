import joblib
import pandas as pd

from churn_mlops.features import add_engineered_features, apply_encoders, apply_scaler


def load_artifacts(model_path="models/rf_model.joblib",
                    encoders_path="models/encoders.joblib",
                    scaler_path="models/scaler.joblib"):
    """
    Loads the trained model, fitted encoders, and fitted scaler.
    """
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    return model, encoders, scaler


def preprocess_new_customer(raw_data: dict, encoders: dict, scaler) -> pd.DataFrame:
    """
    Takes a single raw customer record (a dict of actual values, not
    pre-encoded/scaled), runs it through the same preprocessing used
    at training time, and returns a model-ready DataFrame.
    """
    df = pd.DataFrame([raw_data])
    df = add_engineered_features(df)
    df = apply_encoders(df, encoders)
    df = apply_scaler(df, scaler)
    return df


def predict_churn(model, processed_data: pd.DataFrame):
    """
    Given a fitted model and fully processed data, returns the
    predicted class and churn probability.
    """
    processed_data = processed_data[model.feature_names_in_]
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)[:, 1]
    return predictions, probabilities