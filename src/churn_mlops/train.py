import pandas as pd
import joblib
from churn_mlops.data import get_raw_data
from churn_mlops.clean import drop_leaky_columns
from churn_mlops.features import add_engineered_features, encode_categorical_features, scale_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def prepare_data() -> pd.DataFrame:
    """
    Runs the full data preparation pipeline: downloads and loads the raw
    data, drops leaky columns, adds engineered features, encodes
    categorical columns, and scales continuous features.
    """
    df = get_raw_data()
    df = drop_leaky_columns(df)
    df = add_engineered_features(df)
    df = encode_categorical_features(df)
    df = scale_features(df)
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, model_path: str = "models/rf_model.joblib"):
    """
    Trains a Random Forest classifier and saves it to disk.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    return model

def main():
    df = prepare_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    model = train_random_forest(X_train, y_train)
    print("Model trained and saved to models/rf_model.joblib")


if __name__ == "__main__":
    main()