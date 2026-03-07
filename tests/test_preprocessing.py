"""
tests/test_preprocessing.py
Unit tests for the preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.pipeline import FraudPreprocessor
from src.preprocessing.validator import TransactionRequest


# ── FraudPreprocessor ─────────────────────────────────────────────────
def _make_df(n_legit=1000, n_fraud=20):
    rng = np.random.default_rng(42)
    legit = pd.DataFrame(rng.standard_normal((n_legit, 30)), columns=[f"V{i}" for i in range(29)] + ["Amount"])
    legit["Class"] = 0
    fraud = pd.DataFrame(rng.standard_normal((n_fraud, 30)), columns=[f"V{i}" for i in range(29)] + ["Amount"])
    fraud["Class"] = 1
    return pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)


def test_fit_transform_shapes():
    df = _make_df()
    pp = FraudPreprocessor(use_smote=True)
    X_train, X_test, y_train, y_test = pp.fit_transform(df)
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_smote_balances_classes():
    df = _make_df()
    pp = FraudPreprocessor(use_smote=True)
    _, _, y_train, _ = pp.fit_transform(df)
    unique, counts = np.unique(y_train, return_counts=True)
    # After SMOTE both classes should be roughly equal
    assert counts[0] > 0 and counts[1] > 0


def test_transform_before_fit_raises():
    pp = FraudPreprocessor()
    with pytest.raises(RuntimeError):
        pp.transform(np.zeros((1, 30)))


# ── Validator ──────────────────────────────────────────────────────────
def test_valid_transaction():
    tx = TransactionRequest(features=[0.1] * 29, amount=100.0)
    assert len(tx.features) == 29


def test_invalid_feature_length():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TransactionRequest(features=[0.1] * 10, amount=100.0)


def test_negative_amount():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TransactionRequest(features=[0.1] * 29, amount=-5.0)
