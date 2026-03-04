# Dynamic Pricing Engine

An **AI-powered Dynamic Pricing Optimization Engine** for retail that predicts product demand, finds the revenue-maximising price, and explains its reasoning in plain English.

Built with **FastAPI**, **Streamlit**, **scikit-learn**, **XGBoost**, **TensorFlow/Keras**, **Q-Learning**, and **Google Gemini (via LangChain)**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Dataset](#dataset)
4. [Models & Pipeline](#models--pipeline)
5. [Reinforcement Learning Optimizer](#reinforcement-learning-optimizer)
6. [LLM Explanation Layer](#llm-explanation-layer)
7. [Folder Structure](#folder-structure)
8. [Getting Started](#getting-started)
9. [Running Locally](#running-locally)
10. [API Reference](#api-reference)
11. [Deploying to Production](#deploying-to-production)
12. [FAQ](#faq)

---

## Project Overview

Retail pricing is a balancing act — set the price too high and demand drops; set it too low and you leave money on the table. This engine automates that decision by:

1. **Predicting demand** for a product using an ensemble of four ML/DL models.
2. **Optimising the price** with a Q-Learning reinforcement-learning agent that maximises expected revenue.
3. **Explaining the recommendation** in natural language via Google Gemini so non-technical stakeholders can understand and trust the system.

---

## System Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────────┐
│  Streamlit   │────▶│                FastAPI Backend                   │
│  Dashboard   │◀────│                                                  │
└──────────────┘     │  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
                     │  │  Demand   │  │ Q-Learning│  │   Gemini     │  │
                     │  │ Predictor │  │ Optimizer │  │  Explainer   │  │
                     │  └────┬─────┘  └─────┬─────┘  └──────┬───────┘  │
                     │       │              │               │          │
                     │  ┌────▼──────────────▼───┐    ┌──────▼───────┐  │
                     │  │  Ensemble Model        │    │  LangChain   │  │
                     │  │  RF · XGB · LSTM · MLP │    │  + Gemini    │  │
                     │  └────────────────────────┘    └──────────────┘  │
                     └──────────────────────────────────────────────────┘
```

---

## Dataset

**Retail Price Optimization Dataset** from Kaggle:
<https://www.kaggle.com/datasets/suddharshan/retail-price-optimization>

Download the CSV and place it at:

```
data/retail_price_dataset.csv
```

Key columns: `unit_price`, `qty`, `freight_price`, `comp_1`–`comp_3`, `product_score`, `customers`, `lag_price`, `product_category_name`, etc.

---

## Models & Pipeline

### Preprocessing (`src/preprocessing.py`)
- Loads raw CSV
- Fills missing values (median for numbers, mode for categories)
- Label-encodes categorical columns
- Standardises numerical columns with `StandardScaler`

### Feature Engineering (`src/feature_engineering.py`)
Creates derived features:
| Feature | Formula |
|---------|---------|
| `price_diff` | `unit_price − comp_1` |
| `inventory_ratio` | `qty / max(qty)` |
| `day_of_week` | extracted from `month_year` |
| `month` | extracted from `month_year` |
| `promotion_flag` | `1` if `unit_price < lag_price` else `0` |
| `price_comp_ratio` | `unit_price / comp_1` |
| `log_qty` | `log(1 + qty)` |

### Training (`src/train_models.py`)
Trains four models and saves them to `/models`:

| Model | Library | File |
|-------|---------|------|
| Random Forest | scikit-learn | `random_forest.joblib` |
| XGBoost | xgboost | `xgboost_model.joblib` |
| LSTM | TensorFlow/Keras | `lstm_model.keras` |
| MLP | TensorFlow/Keras | `mlp_model.keras` |

### Ensemble (`src/ensemble_model.py`)
Combines predictions with fixed weights:

```
final = 0.25×RF + 0.35×XGB + 0.25×LSTM + 0.15×MLP
```

---

## Reinforcement Learning Optimizer

`src/rl/qlearning_agent.py` implements tabular **Q-Learning**:

- **State**: product feature vector
- **Action**: pick a price from a grid (e.g. $50 to $200 in $5 steps)
- **Reward**: `price × predicted_demand`
- **Algorithm**: ε-greedy exploration with decaying ε, Bellman Q-value updates

After training, the agent returns the price that maximises expected revenue.

---

## LLM Explanation Layer

`src/llm/gemini_explainer.py` uses **LangChain** with the **Google Gemini** API to generate a 3–5 sentence explanation of a pricing decision. The prompt is designed for a non-technical audience (e.g. a store manager).

**Requirements:**
- Set the `GEMINI_API_KEY` environment variable with your Google AI Studio key.

---

## Folder Structure

```
dynamic-pricing-engine/
├── data/
│   └── retail_price_dataset.csv      # Kaggle dataset (you provide)
├── notebooks/
│   └── EDA.ipynb                     # Exploratory data analysis
├── models/                           # Saved trained models (auto-generated)
├── src/
│   ├── preprocessing.py              # Data cleaning & encoding
│   ├── feature_engineering.py        # Derived features
│   ├── train_models.py               # Train RF, XGB, LSTM, MLP
│   ├── ensemble_model.py             # Weighted ensemble predictor
│   ├── demand_predictor.py           # High-level prediction interface
│   ├── rl/
│   │   └── qlearning_agent.py        # Q-Learning price optimizer
│   ├── api/
│   │   └── main.py                   # FastAPI backend
│   ├── dashboard/
│   │   └── app.py                    # Streamlit frontend
│   ├── llm/
│   │   └── gemini_explainer.py       # Gemini explanation layer
│   └── utils/
│       └── config.py                 # Central configuration
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** (or **conda**)
- A **Google Gemini API key** (free tier from [Google AI Studio](https://aistudio.google.com/))

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd dynamic-pricing-engine
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/suddharshan/retail-price-optimization) and place the CSV at:

```
data/retail_price_dataset.csv
```

### 5. Set environment variables

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-gemini-api-key-here"

# macOS / Linux
export GEMINI_API_KEY="your-gemini-api-key-here"
```

### 6. Train models

```bash
python -m src.train_models
```

This preprocesses the data, trains all four models, and saves them to the `models/` folder.

---

## Running Locally

### Start the FastAPI backend

```bash
uvicorn src.api.main:app --reload --port 10000
```

The API will be available at `http://localhost:10000`. Interactive docs at `http://localhost:10000/docs`.

### Start the Streamlit dashboard

Open a **second terminal** and run:

```bash
streamlit run src/dashboard/app.py
```

The dashboard opens at `http://localhost:8501`.

---

## API Reference

### `GET /health`
Returns `{"status": "ok", "models_loaded": true}`.

### `POST /predict`
**Body:** product attributes (see `PredictRequest` schema in code).

**Response:**
```json
{
  "predicted_demand": 423.5,
  "optimal_price": 110,
  "expected_revenue": 46585.0,
  "individual_predictions": {
    "random_forest": 430.1,
    "xgboost": 418.2,
    "lstm": 425.0,
    "mlp": 420.7
  }
}
```

### `POST /explain`
**Body:**
```json
{
  "price": 110,
  "predicted_demand": 420,
  "competitor_price": 105,
  "inventory": 30
}
```

**Response:**
```json
{
  "explanation": "The recommended price of $110 is set slightly above the competitor's $105 to …"
}
```

### `POST /simulate`
Same body as `/predict`. Returns a list of `{price, predicted_demand, expected_revenue}` for every candidate price.

---

## Deploying to Production

### Backend → Render

1. Push the repo to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Set:
   - **Build command:** `pip install -r requirements.txt && python -m src.train_models`
   - **Start command:** `uvicorn src.api.main:app --host 0.0.0.0 --port 10000`
4. Add environment variable `GEMINI_API_KEY`.
5. Upload `data/retail_price_dataset.csv` to the repo (or use Render Disks for large files).

### Frontend → Vercel

> **Note:** Vercel natively supports static sites and serverless functions.
> For Streamlit you can use the [Streamlit Community Cloud](https://streamlit.io/cloud) (free) as an alternative.

**Using Streamlit Community Cloud:**
1. Push to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Select the repo, branch, and file `src/dashboard/app.py`.
4. Add secret `GEMINI_API_KEY` in the Streamlit secrets manager.
5. Update `API_URL` in `app.py` to point to your Render backend URL.

---

## FAQ

**Q: Do I need a GPU?**
No. The LSTM and MLP are small enough to train on CPU in a few minutes.

**Q: Can I swap model weights?**
Yes — edit `ENSEMBLE_WEIGHTS` in `src/utils/config.py`.

**Q: What if I don't have a Gemini key?**
The `/predict` and `/simulate` endpoints work without it. Only `/explain` requires the key.

**Q: How do I add new features?**
Add them in `src/feature_engineering.py`, then re-run `python -m src.train_models`.

---

**Built with ❤️ for learning — contributions welcome!**
