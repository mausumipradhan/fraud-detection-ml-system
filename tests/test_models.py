"""
tests/test_models.py
Unit tests for model evaluation utilities.
"""

from unittest.mock import MagicMock

import numpy as np

from src.models.evaluate import evaluate_model


def _make_mock_model(probs):
    m = MagicMock()
    m.predict_proba.return_value = np.column_stack([1 - probs, probs])
    return m


def test_evaluate_returns_dict():
    np.random.seed(42)
    y_test = np.array([0] * 90 + [1] * 10)
    probs = np.where(
        y_test == 1, np.random.uniform(0.7, 1.0, 100), np.random.uniform(0.0, 0.3, 100)
    )
    model = _make_mock_model(probs)
    X_test = np.zeros((100, 29))

    metrics = evaluate_model(model, X_test, y_test)
    assert "roc_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["roc_auc"] <= 1.0


def test_high_quality_model_metrics():
    np.random.seed(0)
    y_test = np.array([0] * 950 + [1] * 50)
    probs = np.where(y_test == 1, 0.95, 0.02)
    model = _make_mock_model(probs)
    X_test = np.zeros((1000, 29))

    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["roc_auc"] >= 0.99
    assert metrics["precision"] >= 0.9
    assert metrics["recall"] >= 0.9
