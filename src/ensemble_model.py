"""
ensemble_model.py — Weighted-average ensemble of the four trained models.

Final prediction formula:
    ŷ = 0.25 × RF  +  0.35 × XGB  +  0.25 × LSTM  +  0.15 × MLP

The weights can be tuned in src/utils/config.py (ENSEMBLE_WEIGHTS).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import joblib
from tensorflow import keras

from src.utils.config import (
    ENSEMBLE_WEIGHTS,
    RF_MODEL_PATH, XGB_MODEL_PATH,
    LSTM_MODEL_PATH, MLP_MODEL_PATH,
)


class EnsembleModel:
    """Load all four models once and expose a unified predict() method."""

    def __init__(self):
        self.rf = joblib.load(RF_MODEL_PATH)
        self.xgb = joblib.load(XGB_MODEL_PATH)
        self.lstm = keras.models.load_model(LSTM_MODEL_PATH)
        self.mlp = keras.models.load_model(MLP_MODEL_PATH)
        self.weights = ENSEMBLE_WEIGHTS
        print("[ensemble] All four models loaded.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the weighted-average prediction for feature matrix X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) — ensemble predictions.
        """
        pred_rf = self.rf.predict(X)
        pred_xgb = self.xgb.predict(X)

        # LSTM needs 3-D input (samples, 1, features)
        X_3d = X.reshape((X.shape[0], 1, X.shape[1]))
        pred_lstm = self.lstm.predict(X_3d, verbose=0).flatten()

        pred_mlp = self.mlp.predict(X, verbose=0).flatten()

        # Weighted average
        final = (
            self.weights["random_forest"] * pred_rf
            + self.weights["xgboost"] * pred_xgb
            + self.weights["lstm"] * pred_lstm
            + self.weights["mlp"] * pred_mlp
        )
        return final

    def predict_individual(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return predictions from every constituent model for transparency."""
        X_3d = X.reshape((X.shape[0], 1, X.shape[1]))
        return {
            "random_forest": self.rf.predict(X),
            "xgboost": self.xgb.predict(X),
            "lstm": self.lstm.predict(X_3d, verbose=0).flatten(),
            "mlp": self.mlp.predict(X, verbose=0).flatten(),
        }
