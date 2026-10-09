# Churn MLOps: Credit Card Customer Churn Prediction API

[![CI](https://github.com/Adeyiwolasuccess/churn-mlops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Adeyiwolasuccess/churn-mlops/actions/workflows/ci.yml)

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
- [Model performance](#model-performance)
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
- **Honest evaluation.** The model is scored on a held-out test set with precision, recall, F1, and ROC-AUC, and the results are reproducible with one command.
- **Tested.** Unit tests cover the cleaning step and API tests cover the prediction endpoint. They run automatically on every push.
- **Containerised.** The Docker image trains its own model during the build, so it does not depend on any files from a developer's laptop.
- **Automated delivery.** GitHub Actions trains the model, runs the tests, and builds the Docker image. Render deploys the service only after CI passes.

## Architecture

```
                     git push to main
                           |
                           v
              +-----------------------------+
              |       GitHub Actions        |
              |                             |
              |  1. test:                   |
              |     - train the model       |
              |     - uv run pytest         |
              |  2. build:                  |
              |     - docker build          |
              |       (trains the model)    |
              +--------------+--------------+
                             | CI checks pass
                             v
              +-----------------------------+
              |           Render            |
              |  builds the Dockerfile,     |
              |  trains the model,          |
              |  deploys the container      |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              |   FastAPI  POST /predict    |
              |   encoders -> scaler ->     |
              |   Random Forest             |
              +-----------------------------+
```

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.13 |
| Package management | [uv](https://docs.astral.sh/uv/) |
| Data and ML | pandas, NumPy, scikit-learn, joblib |
| Data source | Kaggle via `kagglehub` |
| API | FastAPI, Pydantic, Uvicorn |
| Testing | pytest, FastAPI `TestClient` |
| Containers | Docker |
| CI | GitHub Actions |
| Hosting | Render |

## Project structure

```
churn-mlops/
├── .github/workflows/ci.yml   # CI pipeline: train and test, then Docker build
├── src/churn_mlops/
│   ├── data.py                # Download the dataset, load to DataFrame and SQLite
│   ├── clean.py               # Drop the leaky Naive Bayes columns
│   ├── features.py            # Feature engineering, encoders, scaler
│   ├── train.py               # End-to-end training script
│   ├── evaluate.py            # Scores the saved model on the held-out test set
│   ├── predict.py             # Load artifacts, preprocess new customers, predict
│   └── api.py                 # FastAPI application
├── tests/
│   ├── test_clean.py          # Unit test for the cleaning step
│   └── test_api.py            # API tests for the health check and /predict
├── Dockerfile                 # Builds the image and trains the model inside it
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Locked dependency versions
└── README.md
```

Generated at training time and **not** committed (they are gitignored): `models/` and `churn.db`.

## The data

- **Dataset:** [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) (BankChurners) from Kaggle, roughly 10,000 customers.
- **Target:** `Attrition_Flag`, mapped to `1` for "Attrited Customer" (churned) and `0` for "Existing Customer". About 16% of customers churn.
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

## Model performance

Evaluated on a held-out test set of 2,026 customers (20% of the data, stratified split, `random_state=42`): 325 churned and 1,701 existing. Reproduce with:

```bash
uv run python -m churn_mlops.evaluate
```

| Metric | Value |
| --- | --- |
| Accuracy | 0.961 |
| Precision | 0.927 |
| Recall | 0.822 |
| F1 score | 0.871 |
| ROC-AUC | 0.989 |

Confusion matrix at the default 0.5 threshold:

| | Predicted: stays | Predicted: churns |
| --- | --- | --- |
| **Actually stays** | 1,680 | 21 |
| **Actually churns** | 58 | 267 |

**How to read this**

- About 16% of customers churn, so a model that always predicted "stays" would already score roughly 84% accuracy. Precision, recall, and ROC-AUC say more than accuracy here.
- **Recall of 0.82:** the model catches 267 of the 325 customers who actually churned and misses 58.
- **Precision of 0.93:** when the model flags a customer as likely to churn, it is right about 93% of the time (21 false alarms against 267 correct flags).
- **ROC-AUC of 0.989:** the model ranks churners above non-churners almost perfectly across all thresholds.
- **The threshold is a business choice.** Lowering it below 0.5 would catch more churners (higher recall) at the cost of more false alarms (lower precision). The right balance depends on what a retention offer costs compared with a lost customer.

**Caveats**

- The encoders and scaler are fitted on the full dataset before the split, so these scores are slightly optimistic. Fitting them on the training split only is planned.
- The model uses default Random Forest settings, with no hyperparameter tuning or class-imbalance handling, and the test set was not used to tune anything.

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

- `422 Unprocessable Entity`: a field is missing, has the wrong type, or a categorical field has a value outside the allowed list. The response body names the offending field and, for categories, lists the allowed values.

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

### Evaluate the model

```bash
uv run python -m churn_mlops.evaluate
```

Prints accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix for the held-out test set.

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

The suite has 5 tests. Because `api.py` loads the saved model when it is imported, run the training step once before testing on a fresh checkout.

**Cleaning (`tests/test_clean.py`)**

- `test_drop_leaky_columns_removes_only_leaky_columns` builds a small fake DataFrame and checks two things: both leaky columns are gone, and the ordinary columns are still present. Together these catch a function that drops too little and one that drops too much.

**API (`tests/test_api.py`)**

- `test_health_check`: `GET /` returns 200.
- `test_predict_returns_valid_probability`: a valid customer returns 200, a prediction of 0 or 1, and a probability between 0 and 1.
- `test_predict_rejects_unknown_category`: an invalid category such as `"Gender": "X"` returns 422 instead of crashing.
- `test_predict_rejects_missing_field`: a request with a required field removed returns 422.

## Docker

Build and run the image:

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

The Dockerfile installs dependencies with `uv`, then runs `python -m churn_mlops.train` as a build step. The trained model, encoders, and scaler are baked into the image, so a container starts ready to serve and every container built from the same image behaves identically. The build needs internet access to download the dataset.

The base image is the official `python:3.13-slim`, pulled from AWS's public mirror (`public.ecr.aws/docker/library/python`) rather than Docker Hub, which avoids Docker Hub's rate limits and occasional outages on shared CI runners.

## CI/CD

The workflow in `.github/workflows/ci.yml` runs on every push to `main` and on every pull request:

| Job | What it does |
| --- | --- |
| `test` | Installs dependencies with `uv`, trains the model (the API tests need the saved artifacts), then runs `uv run pytest` |
| `build` | Runs only if `test` passes. Builds the Docker image, which also trains the model on a clean machine |

Deployment is handled by Render, which builds the same `Dockerfile` and is configured with **Auto-Deploy: After CI Checks Pass**, so a failing test never reaches production.

## Design decisions

- **Train inside the Docker build.** The model folder is gitignored because model files are build artifacts, not source code. Training during the build makes the image self-contained and reproducible.
- **Save and reuse the preprocessors.** Encoders and the scaler are persisted next to the model so prediction applies exactly the transformations used in training.
- **Small, testable functions.** Cleaning, feature engineering, training, and evaluation are separate functions, which is what makes unit tests easy to write.
- **Typed request validation.** Pydantic `Literal` types reject invalid categories with a clear 422 instead of a server error.
- **Tests that can fail.** Each test was checked by breaking the code on purpose and confirming the test caught it.
- **Lock file committed.** `uv.lock` pins every dependency, so local, CI, and production environments match.
- **Registry-independent builds.** Pulling the base image from a public mirror removes a flaky external dependency from CI.

## Known limitations and future work

- **Preprocessors are fitted on the full dataset before the train/test split.** This lets a small amount of information from the test rows into the encoders and scaler, so the reported scores are slightly optimistic. A stricter setup would fit them on the training split only, retrain, and re-run the evaluation.
- **Default decision threshold and model settings.** The model uses a 0.5 threshold and default Random Forest parameters. Threshold tuning against business costs, class-imbalance handling, and hyperparameter search with cross-validation are natural next steps.
- **Model is retrained on every build.** A model registry (such as MLflow) or versioned artifact storage would separate training from deployment.
- **Production image includes dev dependencies.** Installing with `uv sync --frozen --no-dev` would make the image smaller.
- **Free-tier hosting.** The live service sleeps when idle, so the first request can be slow.
- **Possible additions:** monitoring and logging of predictions, data drift checks, and a scheduled retraining workflow.

## Dataset credit

Data from the [Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) dataset on Kaggle by Sakshi Goyal.