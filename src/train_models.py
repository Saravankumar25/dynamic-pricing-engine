"""
train_models.py — Train four demand-prediction models and persist them.

Models trained:
  1. RandomForestRegressor   → saved as joblib
  2. XGBRegressor            → saved as joblib
  3. LSTM (Keras)            → saved as .keras
  4. MLP  (Keras)            → saved as .keras

Usage:
    python -m src.train_models          # from project root
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras

from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering
from src.utils.config import (
    TARGET_COLUMN, MODEL_DIR,
    RF_MODEL_PATH, XGB_MODEL_PATH,
    LSTM_MODEL_PATH, MLP_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
)


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _prepare_data(df: pd.DataFrame):
    """
    Split the DataFrame into features (X) and target (y),
    then create train/test sets.
    """
    # Drop non-numeric & target to form the feature matrix
    drop_cols = [TARGET_COLUMN]
    for col in df.columns:
        if df[col].dtype == "object":
            drop_cols.append(col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)

    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    print(f"[train] Features: {len(feature_cols)} | Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_cols


# ──────────────────────────────────────────────
#  Model 1: Random Forest
# ──────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    """Fit a Random Forest regressor and save it."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RF_MODEL_PATH)
    print(f"[train] Random Forest saved → {RF_MODEL_PATH}")
    return model


# ──────────────────────────────────────────────
#  Model 2: XGBoost
# ──────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    """Fit an XGBoost regressor and save it."""
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, XGB_MODEL_PATH)
    print(f"[train] XGBoost saved → {XGB_MODEL_PATH}")
    return model


# ──────────────────────────────────────────────
#  Model 3: LSTM
# ──────────────────────────────────────────────

def train_lstm(X_train, y_train, X_test, y_test):
    """
    Build and train an LSTM network.
    LSTM expects 3-D input: (samples, timesteps, features).
    We treat each sample as a single timestep.
    """
    # Reshape to (samples, 1, features) for LSTM compatibility
    X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = keras.Sequential([
        keras.layers.LSTM(64, activation="relu", input_shape=(1, X_train.shape[1]),
                          return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(
        X_train_3d, y_train,
        validation_data=(X_test_3d, y_test),
        epochs=30,
        batch_size=64,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    print(f"[train] LSTM saved → {LSTM_MODEL_PATH}")
    return model


# ──────────────────────────────────────────────
#  Model 4: MLP (Multi-Layer Perceptron)
# ──────────────────────────────────────────────

def train_mlp(X_train, y_train, X_test, y_test):
    """Build and train a standard feed-forward neural network."""
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=64,
        verbose=1,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MLP_MODEL_PATH)
    print(f"[train] MLP saved → {MLP_MODEL_PATH}")
    return model


# ──────────────────────────────────────────────
#  Orchestrator
# ──────────────────────────────────────────────

def train_all_models():
    """Run the full training pipeline for every model."""
    # Step 1 — preprocess and engineer features
    df = run_preprocessing_pipeline(fit=True)
    df = run_feature_engineering(df)

    # Step 2 — prepare train/test split
    X_train, X_test, y_train, y_test, feature_cols = _prepare_data(df)

    # Step 3 — train each model
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    lstm_model = train_lstm(X_train, y_train, X_test, y_test)
    mlp_model = train_mlp(X_train, y_train, X_test, y_test)

    # Quick sanity-check scores on the test set
    from sklearn.metrics import mean_absolute_error, r2_score

    for name, model in [("RF", rf_model), ("XGB", xgb_model)]:
        preds = model.predict(X_test)
        print(f"  {name}  MAE={mean_absolute_error(y_test, preds):.4f}  "
              f"R²={r2_score(y_test, preds):.4f}")

    X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    for name, model, data in [("LSTM", lstm_model, X_test_3d), ("MLP", mlp_model, X_test)]:
        preds = model.predict(data, verbose=0).flatten()
        print(f"  {name} MAE={mean_absolute_error(y_test, preds):.4f}  "
              f"R²={r2_score(y_test, preds):.4f}")

    print("\n✓ All models trained and saved.")


if __name__ == "__main__":
    train_all_models()
