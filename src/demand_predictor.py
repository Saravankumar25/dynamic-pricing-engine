"""
demand_predictor.py — High-level interface for demand prediction.

This module ties preprocessing, feature engineering, and the ensemble
model together into a single callable that the API and dashboard import.

Usage:
    from src.demand_predictor import DemandPredictor
    predictor = DemandPredictor()
    result = predictor.predict(input_dict)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib

from src.ensemble_model import EnsembleModel
from src.feature_engineering import run_feature_engineering
from src.utils.config import (
    SCALER_PATH, LABEL_ENCODERS_PATH, FEATURE_COLUMNS_PATH,
)


class DemandPredictor:
    """Stateful predictor — loads models and transformers once on init."""

    def __init__(self):
        self.ensemble = EnsembleModel()
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        print("[demand_predictor] Ready.")

    def _preprocess_input(self, data: dict) -> np.ndarray:
        """
        Turn a raw dict of product attributes into a feature vector
        that matches what the models were trained on.
        """
        df = pd.DataFrame([data])

        # Encode categoricals that are present
        for col, enc in self.label_encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col].astype(str))

        # Feature engineering
        df = run_feature_engineering(df)

        # Scale numerical columns that the scaler knows about
        num_cols = [c for c in self.scaler.feature_names_in_ if c in df.columns]
        if num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols])

        # Align columns with training set; fill any missing ones with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_columns].values.astype(np.float32)
        return X

    def predict(self, data: dict) -> dict:
        """
        Predict demand given a product-attribute dictionary.

        Returns
        -------
        dict with keys: predicted_demand, individual_predictions
        """
        X = self._preprocess_input(data)
        ensemble_pred = self.ensemble.predict(X)
        individual = self.ensemble.predict_individual(X)

        return {
            "predicted_demand": float(ensemble_pred[0]),
            "individual_predictions": {
                k: float(v[0]) for k, v in individual.items()
            },
        }
