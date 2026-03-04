"""
feature_engineering.py — Derive new predictive features from the raw dataset.

Features created:
  • price_diff        – gap between our price and the main competitor
  • inventory_ratio   – normalised inventory relative to typical stock
  • day_of_week       – 0 (Mon) to 6 (Sun)
  • month             – 1–12
  • promotion_flag    – binary indicator of an active promo
  • price_comp_ratio  – our price as a fraction of competitor price
  • log_qty           – log-transformed quantity (reduces skew)

All transformations are deterministic so they work identically at
training time and inference time.
"""

import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


def add_price_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Difference between our unit price and the primary competitor."""
    if "comp_1" in df.columns and "unit_price" in df.columns:
        df["price_diff"] = df["unit_price"] - df["comp_1"]
    elif "unit_price" in df.columns:
        df["price_diff"] = 0.0
    return df


def add_inventory_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of current stock to the column maximum.
    Falls back to a constant if the column is missing.
    """
    if "qty" in df.columns:
        max_qty = df["qty"].max()
        df["inventory_ratio"] = df["qty"] / max_qty if max_qty != 0 else 0.0
    else:
        df["inventory_ratio"] = 0.5
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract day_of_week and month from month_year if it exists."""
    if "month_year" in df.columns:
        try:
            dates = pd.to_datetime(df["month_year"], errors="coerce")
            df["day_of_week"] = dates.dt.dayofweek.fillna(0).astype(int)
            df["month"] = dates.dt.month.fillna(1).astype(int)
        except Exception:
            df["day_of_week"] = 0
            df["month"] = 1
    else:
        df["day_of_week"] = 0
        df["month"] = 1
    return df


def add_promotion_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic promotion flag: if lag_price exists and current price is
    lower, assume a promotion is running.
    """
    if "lag_price" in df.columns and "unit_price" in df.columns:
        df["promotion_flag"] = (df["unit_price"] < df["lag_price"]).astype(int)
    else:
        df["promotion_flag"] = 0
    return df


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Supplementary features that improve model accuracy."""
    if "unit_price" in df.columns and "comp_1" in df.columns:
        df["price_comp_ratio"] = df["unit_price"] / df["comp_1"].replace(0, np.nan)
        df["price_comp_ratio"] = df["price_comp_ratio"].fillna(1.0)

    if "qty" in df.columns:
        df["log_qty"] = np.log1p(df["qty"].clip(lower=0))

    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature-engineering steps in sequence."""
    df = add_price_diff(df)
    df = add_inventory_ratio(df)
    df = add_temporal_features(df)
    df = add_promotion_flag(df)
    df = add_extra_features(df)
    print(f"[feature_engineering] Added derived features — new shape {df.shape}")
    return df


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline

    df = run_preprocessing_pipeline()
    df = run_feature_engineering(df)
    print(df.head())
    print("New columns:", [c for c in df.columns if c in [
        "price_diff", "inventory_ratio", "day_of_week",
        "month", "promotion_flag", "price_comp_ratio", "log_qty",
    ]])
