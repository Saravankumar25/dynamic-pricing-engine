"""
preprocessing.py — Data loading, cleaning, encoding, and scaling.

This module handles the first stage of the ML pipeline:
  1. Load the raw CSV dataset
  2. Drop or impute missing values
  3. Encode categorical columns with LabelEncoder
  4. Scale numerical columns with StandardScaler

Functions return cleaned DataFrames and persist fitted transformers
so they can be reused at inference time.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.config import (
    RAW_DATA_PATH, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS,
    SCALER_PATH, LABEL_ENCODERS_PATH, MODEL_DIR,
)


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the retail price CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[preprocessing] Loaded {len(df)} rows from {path}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Numerical columns → median (robust to outliers)
      - Categorical columns → mode (most-frequent value)
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"[preprocessing] Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")
    return df


def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode every categorical column that exists in the DataFrame.

    Parameters
    ----------
    fit : bool
        If True, fit new encoders and save them.
        If False, load previously fitted encoders.

    Returns
    -------
    df : DataFrame with encoded columns
    encoders : dict mapping column name → fitted LabelEncoder
    """
    encoders: dict[str, LabelEncoder] = {}

    if not fit:
        encoders = joblib.load(LABEL_ENCODERS_PATH)
        for col, enc in encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col].astype(str))
        return df, encoders

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, LABEL_ENCODERS_PATH)
    print(f"[preprocessing] Encoded {len(encoders)} categorical columns")
    return df, encoders


def scale_numerical(
    df: pd.DataFrame,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardise numerical columns to zero mean and unit variance.

    Parameters
    ----------
    fit : bool
        If True, fit a new scaler and persist it.
        If False, load a previously fitted scaler.
    """
    num_cols = [c for c in NUMERICAL_COLUMNS if c in df.columns]

    if not fit:
        scaler = joblib.load(SCALER_PATH)
        df[num_cols] = scaler.transform(df[num_cols])
        return df, scaler

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[preprocessing] Scaled {len(num_cols)} numerical columns")
    return df, scaler


def run_preprocessing_pipeline(
    path: Path = RAW_DATA_PATH,
    fit: bool = True,
) -> pd.DataFrame:
    """
    End-to-end preprocessing: load → clean → encode → scale.
    Returns a fully prepared DataFrame ready for feature engineering.
    """
    df = load_data(path)
    df = handle_missing_values(df)
    df, _ = encode_categoricals(df, fit=fit)
    df, _ = scale_numerical(df, fit=fit)
    print(f"[preprocessing] Pipeline complete — shape {df.shape}")
    return df


if __name__ == "__main__":
    processed = run_preprocessing_pipeline()
    print(processed.head())
