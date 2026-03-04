"""
gemini_explainer.py — LLM explanation layer using Google Gemini directly.

This module generates a natural-language explanation for a pricing decision.
It does NOT depend on LangChain to avoid dependency conflicts.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import google.generativeai as genai
from src.utils.config import GEMINI_API_KEY, GEMINI_MODEL_NAME


def get_explanation(
    price: float,
    predicted_demand: float,
    competitor_price: float,
    inventory: float,
    expected_revenue: float | None = None,
) -> str:

    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        return (
            "⚠️ GEMINI_API_KEY is not set. "
            "Add it to environment variables to enable explanations."
        )

    if expected_revenue is None:
        expected_revenue = price * predicted_demand

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    prompt = f"""
You are an expert retail pricing analyst.

A machine learning system produced the following pricing decision:

Recommended price: ${price}
Predicted demand: {predicted_demand} units
Competitor price: ${competitor_price}
Inventory level: {inventory} units
Expected revenue: ${expected_revenue}

Explain in 3–5 sentences for a non-technical retail store manager:

1. Why this price is competitive.
2. How demand and inventory influenced the recommendation.
3. What business impact the store should expect.
"""

    response = model.generate_content(prompt)

    return response.text.strip()


if __name__ == "__main__":
    sample = {
        "price": 110,
        "predicted_demand": 420,
        "competitor_price": 105,
        "inventory": 30,
    }

    print(get_explanation(**sample))