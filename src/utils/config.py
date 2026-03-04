"""
config.py — Central configuration for the Dynamic Pricing Engine.

All paths, hyperparameters, feature lists, and model weights live here so
every other module can import a single source of truth.
"""

import os
from pathlib import Path

# ── Project root is two levels above this file (src/utils/config.py) ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data paths ──
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "retail_price_dataset.csv"

# ── Trained-model storage ──
MODEL_DIR = PROJECT_ROOT / "models"
RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.keras"
MLP_MODEL_PATH = MODEL_DIR / "mlp_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"

# ── Feature engineering ──
CATEGORICAL_COLUMNS = ["product_category_name", "month_year", "weekday"]
NUMERICAL_COLUMNS = [
    "qty", "total_price", "freight_price", "unit_price",
    "product_name_lenght", "product_description_lenght",
    "product_photos_qty", "product_weight_g", "product_score",
    "customers", "comp_1", "ps1", "fp1",
    "comp_2", "ps2", "fp2",
    "comp_3", "ps3", "fp3",
    "lag_price",
]
TARGET_COLUMN = "unit_price"

# ── Ensemble weights — must sum to 1.0 ──
ENSEMBLE_WEIGHTS = {
    "random_forest": 0.25,
    "xgboost": 0.35,
    "lstm": 0.25,
    "mlp": 0.15,
}

# ── Q-Learning hyperparameters ──
QL_PRICE_MIN = 50
QL_PRICE_MAX = 200
QL_PRICE_STEP = 5
QL_EPISODES = 500
QL_ALPHA = 0.1          # learning rate
QL_GAMMA = 0.95         # discount factor
QL_EPSILON_START = 1.0   # initial exploration rate
QL_EPSILON_MIN = 0.01
QL_EPSILON_DECAY = 0.995

# ── Gemini / LLM ──
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# ── FastAPI ──
API_HOST = "0.0.0.0"
API_PORT = 10000
