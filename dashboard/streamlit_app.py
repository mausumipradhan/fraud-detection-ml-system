"""
dashboard/streamlit_app.py
Real-time monitoring dashboard for the fraud detection system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import random

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

API_URL = "http://localhost:5000"


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.markdown("Real-time monitoring dashboard")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

st.title("🛡️ Real-Time Credit Card Fraud Detection")


# ── Health check ──────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    st.sidebar.success(f"API Online — Model: {health.get('model', 'N/A')}")
except Exception:
    st.sidebar.error("API Offline")

# ── Top metrics ───────────────────────────────────────────────────────
try:
    metrics = requests.get(f"{API_URL}/metrics", timeout=2).json()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions Processed", metrics.get("transactions_processed", 0))
    col2.metric("Recent Fraud Rate", f"{metrics.get('recent_fraud_rate', 0):.2%}")
    col3.metric("Model", metrics.get("model", "N/A").upper())
    col4.metric("Drift Status", metrics.get("drift_status", "N/A").upper())
except Exception:
    st.warning("Could not fetch metrics from API.")


# ── Simulate transactions section ─────────────────────────────────────
st.subheader("📊 Transaction Simulator")
st.markdown("Generate synthetic transactions and submit to the API.")

n = st.number_input("Number of transactions to simulate", 1, 100, 10)

if st.button("🚀 Run Simulation"):
    results = []
    progress = st.progress(0)

    for i in range(n):
        features = np.random.randn(29).tolist()
        payload = {"features": features, "amount": abs(np.random.randn() * 100)}

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5).json()
            results.append({
                "Transaction": i + 1,
                "Fraud Probability": resp.get("fraud_probability", 0),
                "Is Fraud": "🚨 Fraud" if resp.get("is_fraud") else "✅ Legit",
                "Confidence": resp.get("confidence", "N/A"),
                "Timestamp": resp.get("timestamp", ""),
            })
        except Exception as e:
            results.append({"Transaction": i + 1, "Error": str(e)})

        progress.progress((i + 1) / n)

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    # Probability histogram
    probs = df["Fraud Probability"].dropna()
    fig = px.histogram(
        probs, nbins=20, title="Fraud Probability Distribution",
        labels={"value": "Fraud Probability", "count": "Count"},
        color_discrete_sequence=["#E63946"],
    )
    fig.add_vline(x=threshold, line_dash="dash", annotation_text="Threshold")
    st.plotly_chart(fig, use_container_width=True)


# ── Manual prediction ─────────────────────────────────────────────────
st.subheader("🔍 Manual Prediction")
with st.expander("Predict a single transaction"):
    amount = st.number_input("Amount ($)", 0.0, 10000.0, 150.0)
    features_input = st.text_area(
        "Enter 29 feature values (space-separated)",
        value=" ".join([str(round(random.gauss(0, 1), 3)) for _ in range(29)])
    )

    if st.button("Predict"):
        try:
            features = list(map(float, features_input.strip().split()))
            payload = {"features": features, "amount": amount}
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5).json()

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Verdict", "🚨 FRAUD" if resp["is_fraud"] else "✅ LEGIT")
            col_b.metric("Probability", f"{resp['fraud_probability']:.4f}")
            col_c.metric("Confidence", resp["confidence"])

            st.markdown("**Top Contributing Features:**")
            feat_df = pd.DataFrame(resp["explanation"]["top_features"])
            feat_df["direction"] = feat_df["impact"].apply(
                lambda x: "Increases Risk ↑" if x > 0 else "Decreases Risk ↓"
            )
            fig = px.bar(
                feat_df.head(10), x="impact", y="feature", orientation="h",
                color="direction",
                color_discrete_map={"Increases Risk ↑": "#E63946", "Decreases Risk ↓": "#2A9D8F"},
                title="SHAP Feature Contributions",
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")


# ── Auto-refresh ──────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
