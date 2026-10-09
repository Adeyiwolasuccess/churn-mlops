from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn_mlops.data import get_raw_data
from churn_mlops.clean import drop_leaky_columns
from churn_mlops.features import (
    add_engineered_features, fit_encoders, apply_encoders,
    fit_scaler, apply_scaler
)


def prepare_data() -> pd.DataFrame:
    """
    Downloads and loads the raw data, drops leaky columns, and adds
    engineered features. Encoding and scaling happen separately, since
    those steps need to be fit and saved for reuse at prediction time.
    """
    df = get_raw_data()
    df = drop_leaky_columns(df)
    df = add_engineered_features(df)
    return df


def get_train_test_split(df: pd.DataFrame):
    """
    Maps the target to 0/1, builds X and y, and returns a stratified
    80/20 train/test split.
    """
    df['Attrition_Flag'] = df['Attrition_Flag'].map(
        {'Attrited Customer': 1, 'Existing Customer': 0}
    )
    X = df.drop(['Attrition_Flag', 'CLIENTNUM'], axis=1)
    y = df['Attrition_Flag']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_random_forest(X_train, y_train, model_path: str = "models/rf_model.joblib"):
    """
    Trains a Random Forest classifier and saves it to disk.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model


def main():
    # Make sure the output folder exists (it's gitignored, so a fresh
    # checkout or Docker build won't have it)
    Path("models").mkdir(exist_ok=True)

    df = prepare_data()

    # Fit encoders and scaler on the full dataset, save them for reuse
    encoders = fit_encoders(df)
    scaler = fit_scaler(df)
    joblib.dump(encoders, "models/encoders.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    # Apply them before splitting and training
    df = apply_encoders(df, encoders)
    df = apply_scaler(df, scaler)

    X_train, X_test, y_train, y_test = get_train_test_split(df)
    train_random_forest(X_train, y_train)
    print("Model, encoders, and scaler trained and saved to models/")


if __name__ == "__main__":
    main()