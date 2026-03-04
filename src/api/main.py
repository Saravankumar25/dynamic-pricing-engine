"""
main.py — FastAPI backend for the Dynamic Pricing Engine.

Endpoints
---------
POST /predict    → predicted demand, optimal price, expected revenue
POST /explain    → natural-language pricing explanation (Gemini)
POST /simulate   → price vs demand vs revenue table for charting
GET  /health     → simple liveness check
"""

import sys
from pathlib import Path

# Ensure the project root is on the Python path so all src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.demand_predictor import DemandPredictor
from src.rl.qlearning_agent import QLearningPriceOptimizer
from src.llm.gemini_explainer import get_explanation

# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Dynamic Pricing Engine API",
    description="AI-powered demand prediction and price optimisation.",
    version="1.0.0",
)

# Allow the Streamlit frontend (and any other origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models once at startup ──────────────────────────────────────
predictor: DemandPredictor | None = None


@app.on_event("startup")
def load_models():
    """Load all ML models into memory when the server starts."""
    global predictor
    try:
        predictor = DemandPredictor()
        print("[api] Models loaded successfully.")
    except Exception as e:
        print(f"[api] ⚠️  Could not load models: {e}")
        print("[api] Endpoints will return errors until models are trained.")


# ── Request / Response schemas ───────────────────────────────────────

class PredictRequest(BaseModel):
    """Input payload for the /predict and /simulate endpoints."""
    product_category_name: str = Field("electronics", description="Product category")
    qty: float = Field(100, description="Current inventory quantity")
    unit_price: float = Field(100, description="Current unit price")
    comp_1: float = Field(95, description="Primary competitor price")
    freight_price: float = Field(10, description="Freight / shipping cost")
    product_weight_g: float = Field(500, description="Product weight in grams")
    product_score: float = Field(4.0, description="Customer review score (0-5)")
    customers: float = Field(50, description="Number of recent customers")
    lag_price: float = Field(105, description="Previous period price")
    total_price: float = Field(110, description="Total listed price")
    product_name_lenght: float = Field(30, description="Length of product name")
    product_description_lenght: float = Field(200, description="Length of product description")
    product_photos_qty: float = Field(3, description="Number of product photos")
    comp_2: float = Field(98, description="Competitor 2 price")
    ps2: float = Field(4.0, description="Competitor 2 product score")
    fp2: float = Field(10, description="Competitor 2 freight price")
    comp_3: float = Field(100, description="Competitor 3 price")
    ps3: float = Field(3.8, description="Competitor 3 product score")
    fp3: float = Field(12, description="Competitor 3 freight price")
    ps1: float = Field(4.2, description="Competitor 1 product score")
    fp1: float = Field(9, description="Competitor 1 freight price")
    month_year: str = Field("2024-01", description="Month-year string")
    weekday: str = Field("Monday", description="Day of week")


class PredictResponse(BaseModel):
    predicted_demand: float
    optimal_price: float
    expected_revenue: float
    individual_predictions: dict


class ExplainRequest(BaseModel):
    price: float = Field(..., description="Recommended price")
    predicted_demand: float = Field(..., description="Predicted demand units")
    competitor_price: float = Field(..., description="Main competitor price")
    inventory: float = Field(..., description="Current inventory")
    expected_revenue: float | None = Field(None, description="Expected revenue")


class ExplainResponse(BaseModel):
    explanation: str


class SimulateResponse(BaseModel):
    simulations: list[dict]


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict demand using the ensemble model, then run Q-Learning to
    find the optimal price and expected revenue.
    """
    if predictor is None:
        raise HTTPException(503, "Models not loaded. Train models first.")

    # Build a plain dict from the request
    data = req.model_dump()

    # Step 1 — ensemble demand prediction
    result = predictor.predict(data)
    predicted_demand = result["predicted_demand"]

    # Step 2 — Q-Learning price optimisation
    # The demand function returns the ensemble prediction for a given price
    def demand_fn(price, features):
        modified = data.copy()
        modified["unit_price"] = price
        return predictor.predict(modified)["predicted_demand"]

    features = np.zeros(1)  # placeholder — agent hashes internally
    agent = QLearningPriceOptimizer(demand_fn, features)
    agent.train()
    optimal = agent.get_optimal_price()

    return PredictResponse(
        predicted_demand=round(predicted_demand, 2),
        optimal_price=optimal["optimal_price"],
        expected_revenue=round(optimal["expected_revenue"], 2),
        individual_predictions=result["individual_predictions"],
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """Ask Gemini to explain a pricing decision in plain English."""
    explanation = get_explanation(
        price=req.price,
        predicted_demand=req.predicted_demand,
        competitor_price=req.competitor_price,
        inventory=req.inventory,
        expected_revenue=req.expected_revenue,
    )
    return ExplainResponse(explanation=explanation)


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: PredictRequest):
    """
    Sweep over candidate prices and return demand + revenue for each
    so the frontend can plot price-vs-demand and price-vs-revenue curves.
    """
    if predictor is None:
        raise HTTPException(503, "Models not loaded. Train models first.")

    data = req.model_dump()

    def demand_fn(price, features):
        modified = data.copy()
        modified["unit_price"] = price
        return predictor.predict(modified)["predicted_demand"]

    features = np.zeros(1)
    agent = QLearningPriceOptimizer(demand_fn, features)
    simulations = agent.simulate_prices()

    return SimulateResponse(simulations=simulations)


# ── Direct-run entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=10000, reload=True)
