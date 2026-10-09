# Churn MLOps: Credit Card Customer Churn Prediction API

[![CI](https://github.com/Adeyiwolasuccess/churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Adeyiwolasuccess/churn-mlops/actions/workflows/ci.yml)

An end-to-end machine learning project that predicts whether a credit card customer is likely to churn, served as a REST API and shipped through an automated CI/CD pipeline.

**Live API:** https://churn-mlops-3r7c.onrender.com
**Interactive docs (Swagger):** https://churn-mlops-3r7c.onrender.com/docs

> The API runs on Render's free tier, which sleeps when idle. The first request after a quiet period can take up to a minute while the service wakes up.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [The data](#the-data)
- [Pipeline details](#pipeline-details)
- [API reference](#api-reference)
- [Getting started](#getting-started)
- [Testing](#testing)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Design decisions](#design-decisions)
- [Known limitations and future work](#known-limitations-and-future-work)

---

## Overview

Customer churn is expensive: keeping an existing customer is usually cheaper than acquiring a new one. This project trains a Random Forest classifier on the BankChurners dataset and exposes it through a FastAPI service. Send in a customer's profile and account activity, and get back a predicted class and a churn probability.

Beyond the model itself, the focus is on the engineering around it:

- **Reproducible training.** A single command downloads the data, cleans it, engineers features, fits the encoders and scaler, trains the model, and saves every artifact.
- **Consistent preprocessing.** The exact encoders and scaler fitted at training time are saved and reused at prediction time, so the API transforms new data the same way the model was trained.
- **Leakage prevention.** Two pre-computed Naive Bayes columns that ship with the dataset leak the target and are dropped before training.
- **Tested.** Unit tests guard the cleaning step and run automatically on every push.
- **Containerised.** The Docker image trains its own model during the build, so it does not depend on any files from a developer's laptop.
- **Automated delivery.** GitHub Actions runs the tests and the Docker build, and Render deploys the service only after CI passes.

## Architecture

```
                     git push to main
                           |
                           v
              +---------------------------+
              |      GitHub Actions       |
              |                           |
              |  1. test:  uv run pytest  |
              |  2. build: docker build   |
              |     (trains the model)    |
              +-------------+-------------+
                            | CI checks pass
                            v
              +---------------------------+
              |          Render           |
              |  builds the Dockerfile,   |
              |  trains the model,        |
              |  deploys the container    |
              +-------------+-------------+
                            |
                            v
              +---------------------------+
              |  FastAPI  /predict        |
              |  encoders -> scaler ->    |
              |  Random Forest            |
              +---------------------------+
```

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.13 |
| Package management | [uv](https://docs.astral.sh/uv/) |
| Data and ML | pandas, NumPy, scikit-learn, joblib |
| Data source | Kaggle via `kagglehub` |
| API | FastAPI, Pydantic, Uvicorn |
| Testing | pytest |
| Containers | Docker |
| CI | GitHub Actions |
| Hosting | Render |

## Project structure

```
churn-mlops/
├── .github/workflows/ci.yml   # CI pipeline: test, then Docker build
├── src/churn_mlops/
│   ├── data.py                # Download the dataset, load to DataFrame and SQLite
│   ├── clean.py               # Drop the leaky Naive Bayes columns
│   ├── features.py            # Feature engineering, encoders, scaler
│   ├── train.py               # End-to-end training script
│   ├── predict.py             # Load artifacts, preprocess new customers, predict
│   └── api.py                 # FastAPI application
├── tests/
│   └── test_clean.py          # Unit tests for the cleaning step
├── Dockerfile                 # Builds the image and trains the model inside it
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions
└── README.md
```

Generated at training time and **not** committed (they are gitignored): `models/` and `churn.db`.

## The data

- **Dataset:** [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) (BankChurners) from Kaggle, roughly 10,000 customers.
- **Target:** `Attrition_Flag`, mapped to `1` for "Attrited Customer" (churned) and `0` for "Existing Customer".
- **Features:** customer demographics (age, gender, dependents, education, marital status, income category), card details (card category, months on book, credit limit), and activity (transaction amounts and counts, revolving balance, utilisation ratio, contact and inactivity counts).
- **Download:** the dataset is fetched automatically by `kagglehub` when training runs. No CSV is stored in the repository.

### Data leakage

The raw file includes two columns whose names begin with `Naive_Bayes_Classifier_...`. They are the output of a previous model trained on this same data and are almost perfectly correlated with the target. Training on them would produce impressive but meaningless results, and they would not exist for a real new customer. `drop_leaky_columns()` in `clean.py` removes them first, and a unit test makes sure that stays true.

## Pipeline details

Running `churn_mlops.train` performs these steps in order:

1. **Load.** `get_raw_data()` downloads the dataset, loads it into a DataFrame, and writes a copy to a local SQLite database (`churn.db`).
2. **Clean.** `drop_leaky_columns()` removes the two leaky columns.
3. **Engineer features.** `add_engineered_features()` derives additional columns from the raw ones (see `features.py` for the exact definitions).
4. **Fit encoders and scaler.** The categorical encoders and numeric scaler are fitted and saved to `models/encoders.joblib` and `models/scaler.joblib` so they can be reused at prediction time.
5. **Transform.** The encoders and scaler are applied to the data.
6. **Split.** The target is mapped to 0/1 and the data is split 80/20, stratified by the target so both sets keep the same churn rate (`random_state=42`).
7. **Train.** A `RandomForestClassifier` with 100 trees (`random_state=42`) is fitted and saved to `models/rf_model.joblib`.

Because every random seed is fixed, the same data always produces the same model.

At prediction time, `predict.py` loads the saved model, encoders, and scaler once at startup. Each request goes through the same engineering, encoding, and scaling steps before the model scores it.

## API reference

Base URL: `https://churn-mlops-3r7c.onrender.com`

### `GET /`

Health check.

```json
{ "status": "Churn prediction API is running" }
```

### `POST /predict`

Accepts a customer's raw features and returns the predicted class and churn probability.

**Request body**

| Field | Type | Allowed values |
| --- | --- | --- |
| `Customer_Age` | number | |
| `Dependent_count` | number | |
| `Months_on_book` | number | |
| `Total_Relationship_Count` | number | |
| `Months_Inactive_12_mon` | number | |
| `Contacts_Count_12_mon` | number | |
| `Credit_Limit` | number | |
| `Total_Revolving_Bal` | number | |
| `Avg_Open_To_Buy` | number | |
| `Total_Amt_Chng_Q4_Q1` | number | |
| `Total_Trans_Amt` | number | |
| `Total_Trans_Ct` | number | |
| `Total_Ct_Chng_Q4_Q1` | number | |
| `Avg_Utilization_Ratio` | number | |
| `Gender` | string | `M`, `F` |
| `Education_Level` | string | `Uneducated`, `High School`, `College`, `Graduate`, `Post-Graduate`, `Doctorate`, `Unknown` |
| `Marital_Status` | string | `Married`, `Single`, `Divorced`, `Unknown` |
| `Income_Category` | string | `Less than $40K`, `$40K - $60K`, `$60K - $80K`, `$80K - $120K`, `$120K +`, `Unknown` |
| `Card_Category` | string | `Blue`, `Silver`, `Gold`, `Platinum` |

**Example request**

```bash
curl -X POST "https://churn-mlops-3r7c.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "Avg_Utilization_Ratio": 0.061
  }'
```

**Example response**

```json
{
  "prediction": 0,
  "churn_probability": 0.03
}
```

- `prediction`: `1` if the customer is predicted to churn, `0` otherwise.
- `churn_probability`: the model's estimated probability of churn, between 0 and 1.

**Errors**

- `422 Unprocessable Entity`: a field is missing, has the wrong type, or a categorical field has a value outside the allowed list. The response body names the offending field.

## Getting started

### Prerequisites

- Python 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Internet access (the dataset is downloaded from Kaggle on first run)

### Install

```bash
git clone https://github.com/Adeyiwolasuccess/churn-mlops.git
cd churn-mlops
uv sync --all-groups
```

### Train the model

```bash
uv run python -m churn_mlops.train
```

This creates the `models/` folder containing `rf_model.joblib`, `encoders.joblib`, and `scaler.joblib`, and writes `churn.db`.

### Run the API locally

```bash
uv run uvicorn churn_mlops.api:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints in Swagger.

> The API loads the model files when it starts, so run the training step first.

## Testing

```bash
uv run pytest
```

The current test suite covers the cleaning step:

- `test_drop_leaky_columns_removes_only_leaky_columns` builds a small fake DataFrame and checks two things: both leaky columns are gone, and the ordinary columns are still present. Together these catch a function that drops too little and one that drops too much.

## Docker

Build and run the image:

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

The Dockerfile installs dependencies with `uv`, then runs `python -m churn_mlops.train` as a build step. The trained model, encoders, and scaler are baked into the image, so a container starts ready to serve and every container built from the same image behaves identically. The build needs internet access to download the dataset.

## CI/CD

The workflow in `.github/workflows/ci.yml` runs on every push to `main` and on every pull request:

| Job | What it does |
| --- | --- |
| `test` | Installs dependencies with `uv` and runs `uv run pytest` |
| `build` | Runs only if `test` passes. Builds the Docker image, which also trains the model on a clean machine |

Deployment is handled by Render, which builds the same `Dockerfile` and is configured with **Auto-Deploy: After CI Checks Pass**, so a failing test never reaches production.

## Design decisions

- **Train inside the Docker build.** The model folder is gitignored because model files are build artifacts, not source code. Training during the build makes the image self-contained and reproducible.
- **Save and reuse the preprocessors.** Encoders and the scaler are persisted next to the model so prediction applies exactly the transformations used in training.
- **Small, testable functions.** Cleaning, feature engineering, and training are separate functions, which is what makes unit tests easy to write.
- **Typed request validation.** Pydantic models with `Literal` types reject invalid categories with a clear 422 instead of a server error.
- **Lock file committed.** `uv.lock` pins every dependency, so local, CI, and production environments match.

## Known limitations and future work

- **Preprocessors are fitted on the full dataset before the train/test split.** This lets a small amount of information from the test rows into the encoders and scaler. A stricter setup would fit them on the training split only.
- **No model evaluation report yet.** Add precision, recall, F1, and ROC-AUC on the held-out test set (churn is the minority class, so accuracy alone is misleading), and consider hyperparameter tuning and class-imbalance handling.
- **Thin test coverage.** Tests cover the cleaning step only. Good next additions are a prediction smoke test (valid input returns a probability between 0 and 1) and API tests using FastAPI's `TestClient`.
- **Model is retrained on every build.** A model registry (such as MLflow) or versioned artifact storage would separate training from deployment.
- **Free-tier hosting.** The live service sleeps when idle, so the first request can be slow.
- **Possible additions:** monitoring and logging of predictions, data drift checks, and a scheduled retraining workflow.

## Dataset credit

Data from the [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) dataset on Kaggle by Sakshi Goyal.