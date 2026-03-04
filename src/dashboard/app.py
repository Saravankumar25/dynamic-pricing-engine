"""
app.py — Streamlit dashboard for the Dynamic Pricing Engine.

Pages
-----
1. Pricing Dashboard   — enter product info → get demand, optimal price, revenue
2. Price Simulation    — interactive price-vs-demand and price-vs-revenue charts
3. AI Explanation      — ask Gemini why a price was recommended

Run:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────
API_URL = "http://localhost:10000"

st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a modern look ─────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar inputs (shared across pages) ─────────────────────────────
st.sidebar.markdown("## Product Parameters")
product_category = st.sidebar.selectbox(
    "Product Category",
    ["electronics", "furniture", "clothing", "sports", "toys", "health", "auto"],
)
unit_price = st.sidebar.number_input("Current Unit Price ($)", 10.0, 500.0, 100.0, step=5.0)
comp_price = st.sidebar.number_input("Competitor Price ($)", 10.0, 500.0, 95.0, step=5.0)
inventory = st.sidebar.number_input("Inventory (qty)", 1, 10000, 100, step=10)
freight = st.sidebar.number_input("Freight Price ($)", 0.0, 100.0, 10.0, step=1.0)
product_score = st.sidebar.slider("Product Score", 0.0, 5.0, 4.0, 0.1)
customers = st.sidebar.number_input("Recent Customers", 0, 5000, 50, step=10)
lag_price = st.sidebar.number_input("Previous Price ($)", 10.0, 500.0, 105.0, step=5.0)
promotion = st.sidebar.checkbox("Promotion Active")


def _build_payload() -> dict:
    """Assemble the API request body from sidebar inputs."""
    return {
        "product_category_name": product_category,
        "qty": inventory,
        "unit_price": unit_price,
        "comp_1": comp_price,
        "freight_price": freight,
        "product_weight_g": 500,
        "product_score": product_score,
        "customers": customers,
        "lag_price": lag_price,
        "total_price": unit_price + freight,
        "product_name_lenght": 30,
        "product_description_lenght": 200,
        "product_photos_qty": 3,
        "comp_2": comp_price * 1.02,
        "ps2": 4.0,
        "fp2": 10,
        "comp_3": comp_price * 1.05,
        "ps3": 3.8,
        "fp3": 12,
        "ps1": 4.2,
        "fp1": 9,
        "month_year": "2024-01",
        "weekday": "Monday",
    }


# ── Page tabs ────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Pricing Dashboard",
    "📈 Price Simulation",
    "🤖 AI Explanation",
])

# ────────────────────────────────────────────────────────────────────
#  TAB 1: Pricing Dashboard
# ────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="main-header">Dynamic Pricing Dashboard</p>', unsafe_allow_html=True)
    st.write("Enter product parameters in the sidebar, then click **Get Optimal Price**.")

    if st.button("🔍 Get Optimal Price", key="predict_btn"):
        with st.spinner("Running demand prediction & price optimisation..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=_build_payload(), timeout=120)
                resp.raise_for_status()
                data = resp.json()

                # Display key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Demand", f"{data['predicted_demand']:,.2f} units")
                with col2:
                    st.metric("Optimal Price", f"${data['optimal_price']:,.2f}")
                with col3:
                    st.metric("Expected Revenue", f"${data['expected_revenue']:,.2f}")

                # Individual model predictions
                st.subheader("Individual Model Predictions")
                model_df = pd.DataFrame(
                    list(data["individual_predictions"].items()),
                    columns=["Model", "Prediction"],
                )
                fig = px.bar(
                    model_df, x="Model", y="Prediction",
                    color="Model",
                    title="Demand Predictions by Model",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Make sure the FastAPI backend is running on port 10000.")
            except Exception as e:
                st.error(f"Error: {e}")

# ────────────────────────────────────────────────────────────────────
#  TAB 2: Price Simulation
# ────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="main-header">Price Simulation</p>', unsafe_allow_html=True)
    st.write("Explore how different prices affect demand and revenue.")

    if st.button("🚀 Run Simulation", key="sim_btn"):
        with st.spinner("Simulating price range..."):
            try:
                resp = requests.post(f"{API_URL}/simulate", json=_build_payload(), timeout=120)
                resp.raise_for_status()
                sims = resp.json()["simulations"]
                sim_df = pd.DataFrame(sims)

                col1, col2 = st.columns(2)

                with col1:
                    fig1 = px.line(
                        sim_df, x="price", y="predicted_demand",
                        title="Price vs Predicted Demand",
                        markers=True,
                    )
                    fig1.update_layout(
                        xaxis_title="Price ($)",
                        yaxis_title="Predicted Demand",
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    fig2 = px.area(
                        sim_df, x="price", y="expected_revenue",
                        title="Price vs Expected Revenue",
                    )
                    fig2.update_layout(
                        xaxis_title="Price ($)",
                        yaxis_title="Expected Revenue ($)",
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Highlight the revenue-maximising price
                best = sim_df.loc[sim_df["expected_revenue"].idxmax()]
                st.success(
                    f"Revenue-maximising price: **${best['price']:.0f}** "
                    f"→ demand {best['predicted_demand']:.0f} units, "
                    f"revenue ${best['expected_revenue']:,.0f}"
                )

                st.dataframe(sim_df, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Start the FastAPI backend first.")
            except Exception as e:
                st.error(f"Error: {e}")

# ────────────────────────────────────────────────────────────────────
#  TAB 3: AI Explanation
# ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="main-header">AI Pricing Explanation</p>', unsafe_allow_html=True)
    st.write(
        "Get a plain-English explanation of a pricing recommendation "
        "powered by Google Gemini."
    )

    with st.form("explain_form"):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            exp_price = st.number_input("Recommended Price ($)", value=110.0, step=5.0)
            exp_demand = st.number_input("Predicted Demand (units)", value=420.0, step=10.0)
        with ecol2:
            exp_comp = st.number_input("Competitor Price ($)", value=105.0, step=5.0)
            exp_inv = st.number_input("Inventory (units)", value=30.0, step=5.0)

        submitted = st.form_submit_button("💡 Explain Pricing Decision")

    if submitted:
        with st.spinner("Asking Gemini..."):
            try:
                payload = {
                    "price": exp_price,
                    "predicted_demand": exp_demand,
                    "competitor_price": exp_comp,
                    "inventory": exp_inv,
                }
                resp = requests.post(f"{API_URL}/explain", json=payload, timeout=30)
                resp.raise_for_status()
                explanation = resp.json()["explanation"]
                st.info(explanation)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Start the FastAPI backend first.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Footer ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Dynamic Pricing Engine v1.0 • Built with Streamlit + FastAPI")
