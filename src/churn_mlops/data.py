"""
data.py

Handles downloading the BankChurners dataset from Kaggle and loading it
into both a pandas DataFrame and a SQLite database.
"""

import os
import sqlite3

import kagglehub
import pandas as pd


def download_dataset() -> str:
    """
    Downloads the BankChurners dataset from Kaggle using kagglehub.
    Returns the local file path to the downloaded CSV.
    """
    path = kagglehub.dataset_download("sakshigoyal7/credit-card-customers")
    csv_file = os.path.join(path, "BankChurners.csv")
    return csv_file


def load_to_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV at the given path into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df


def load_to_sqlite(df: pd.DataFrame, db_path: str = "churn.db", table_name: str = "churn_raw") -> None:
    """
    Pushes a DataFrame into a SQLite database as a table.
    """
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


def get_raw_data() -> pd.DataFrame:
    """
    Full pipeline: download from Kaggle, load into SQLite, and return
    the DataFrame ready for cleaning.
    """
    csv_path = download_dataset()
    df = load_to_dataframe(csv_path)
    load_to_sqlite(df)
    return df