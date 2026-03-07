"""
api/app.py
Flask REST API for real-time credit card fraud detection.
"""

import os
import logging
import yaml
import numpy as np
import joblib
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import ValidationError

from src.preprocessing.validator import TransactionRequest
from src.explainability.shap_explainer import FraudExplainer
from src.monitoring.drift_detector import DriftDetector

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App init ─────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load config ───────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Load model + preprocessor ────────────────────────────────────────
MODEL_DIR = cfg["model"]["save_path"]
MODEL_TYPE = cfg["model"]["type"]
THRESHOLD = cfg["model"]["threshold"]

model = joblib.load(os.path.join(MODEL_DIR, f"{MODEL_TYPE}_model.joblib"))
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))

explainer = FraudExplainer(model, feature_names=preprocessor.feature_names)

# Drift detector is initialised lazily (needs reference data)
drift_detector: DriftDetector | None = None
_transactions_log = []  # in-memory store (replace with DB in prod)


def _confidence(prob: float) -> str:
    if prob < 0.3 or prob > 0.7:
        return "HIGH"
    if prob < 0.45 or prob > 0.55:
        return "MEDIUM"
    return "LOW"


# ── Routes ────────────────────────────────────────────────────────────


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_TYPE, "threshold": THRESHOLD}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict fraud for a single transaction.

    Body JSON:
        {
            "features": [29 floats],
            "amount": float
        }
    """
    try:
        payload = TransactionRequest(**request.get_json(force=True))
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    raw = np.array(payload.features + [payload.amount], dtype=float)
    X = preprocessor.scaler.transform(raw.reshape(1, -1))

    prob = float(model.predict_proba(X)[0, 1])
    is_fraud = prob >= THRESHOLD

    explanation = explainer.explain_instance(X[0])

    # Track for drift monitoring
    _transactions_log.append({"features": raw.tolist(), "prob": prob})
    if drift_detector:
        drift_detector.add_transaction(raw)

    response = {
        "is_fraud": is_fraud,
        "fraud_probability": round(prob, 4),
        "confidence": _confidence(prob),
        "explanation": explanation,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(response), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return model info and drift status."""
    recent_probs = [t["prob"] for t in _transactions_log[-1000:]]
    fraud_rate = sum(p >= THRESHOLD for p in recent_probs) / max(len(recent_probs), 1)

    drift_status = "unknown"
    if drift_detector:
        report = drift_detector.check_drift()
        drift_status = "drifted" if (report and report.overall_drift) else "stable"

    return (
        jsonify(
            {
                "model": MODEL_TYPE,
                "threshold": THRESHOLD,
                "transactions_processed": len(_transactions_log),
                "recent_fraud_rate": round(fraud_rate, 4),
                "drift_status": drift_status,
            }
        ),
        200,
    )


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Predict fraud for a batch of transactions.

    Body JSON:
        {"transactions": [ {"features": [...], "amount": ...}, ... ]}
    """
    data = request.get_json(force=True)
    results = []
    for tx in data.get("transactions", []):
        try:
            payload = TransactionRequest(**tx)
        except ValidationError as e:
            results.append({"error": str(e)})
            continue

        raw = np.array(payload.features + [payload.amount], dtype=float)
        X = preprocessor.scaler.transform(raw.reshape(1, -1))
        prob = float(model.predict_proba(X)[0, 1])
        results.append(
            {
                "is_fraud": prob >= THRESHOLD,
                "fraud_probability": round(prob, 4),
            }
        )

    return jsonify({"results": results}), 200


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        debug=cfg["api"]["debug"],
    )
