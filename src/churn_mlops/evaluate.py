import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_mlops.features import apply_encoders, apply_scaler
from churn_mlops.train import get_train_test_split, prepare_data


def main():
    # Rebuilding the exact same data the model was trained on
    df = prepare_data()

    # Reuse the saved preprocessors (do not refit them here)
    encoders = joblib.load("models/encoders.joblib")
    scaler = joblib.load("models/scaler.joblib")
    df = apply_encoders(df, encoders)
    df = apply_scaler(df, scaler)

    # Same seed and stratification as training, so X_test is the held-out 20%
    _, X_test, _, y_test = get_train_test_split(df)

    model = joblib.load("models/rf_model.joblib")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    print(f"Test set size: {len(y_test)} "
          f"({int(y_test.sum())} churned, {int((y_test == 0).sum())} existing)")
    print()
    print("| Metric    | Value  |")
    print("| --------- | ------ |")
    print(f"| Accuracy  | {accuracy_score(y_test, predictions):.3f}  |")
    print(f"| Precision | {precision_score(y_test, predictions):.3f}  |")
    print(f"| Recall    | {recall_score(y_test, predictions):.3f}  |")
    print(f"| F1 score  | {f1_score(y_test, predictions):.3f}  |")
    print(f"| ROC-AUC   | {roc_auc_score(y_test, probabilities):.3f}  |")
    print()
    print("Confusion matrix (threshold 0.5):")
    print(f"  True negatives:  {tn}   False positives: {fp}")
    print(f"  False negatives: {fn}   True positives:  {tp}")


if __name__ == "__main__":
    main()