"""
preprocessing/validator.py
Pydantic schemas for validating API request payloads.
"""

from pydantic import BaseModel, field_validator
from typing import List


class TransactionRequest(BaseModel):
    """
    Incoming transaction for fraud prediction.

    `features` must contain exactly 28 values (V1–V28 + Amount).
    """

    features: List[float]
    amount: float

    @field_validator("features")
    @classmethod
    def check_feature_length(cls, v: List[float]) -> List[float]:
        if len(v) != 28:
            raise ValueError(f"Expected 29 features (V1–V28 + Amount), got {len(v)}")
        return v

    @field_validator("amount")
    @classmethod
    def check_positive_amount(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Transaction amount must be non-negative.")
        return v


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    confidence: str  # LOW / MEDIUM / HIGH
    explanation: dict
    timestamp: str
