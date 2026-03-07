"""
tests/test_api.py
Unit tests for the Flask API endpoints.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict_proba.return_value = np.array([[0.95, 0.05]])
    return m


@pytest.fixture
def mock_preprocessor():
    p = MagicMock()
    p.scaler.transform.return_value = np.zeros((1, 28))
    p.feature_names = [f"V{i}" for i in range(28)]
    return p


@pytest.fixture
def client(mock_model, mock_preprocessor):
    with patch("joblib.load") as mock_load, patch("builtins.open", MagicMock()), patch(
        "yaml.safe_load",
        return_value={
            "model": {"type": "xgboost", "threshold": 0.5, "save_path": "models/"},
            "api": {"host": "0.0.0.0", "port": 5000, "debug": False},
        },
    ), patch("src.api.app.FraudExplainer") as mock_explainer:

        mock_load.side_effect = [mock_model, mock_preprocessor]

        # mock SHAP explainer
        explainer_instance = mock_explainer.return_value
        explainer_instance.explain_instance.return_value = {"V1": 0.12, "V2": -0.05}

        from src.api.app import app

        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────
def make_payload(prob=0.05):
    return {
        "features": [float(i) * 0.01 for i in range(28)],
        "amount": 150.0,
    }


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


def test_predict_legit(client):
    resp = client.post(
        "/predict",
        data=json.dumps(make_payload()),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "confidence" in data
    assert "explanation" in data


def test_predict_bad_features(client):
    resp = client.post(
        "/predict",
        data=json.dumps({"features": [1.0, 2.0], "amount": 10.0}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_negative_amount(client):
    payload = make_payload()
    payload["amount"] = -50.0
    resp = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_batch_predict(client):
    resp = client.post(
        "/batch_predict",
        data=json.dumps({"transactions": [make_payload(), make_payload()]}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["results"]) == 2
