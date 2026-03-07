"""
models/train.py
Training pipeline for Logistic Regression, Random Forest, and XGBoost.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.evaluate import evaluate_model
from src.preprocessing.pipeline import FraudPreprocessor

logger = logging.getLogger(__name__)


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=100,  # handles class imbalance
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False,
    ),
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path: str = "config.yaml", model_type: str | None = None) -> None:
    cfg = load_config(config_path)
    model_type = model_type or cfg["model"]["type"]

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_type}'. Choose from {list(MODEL_REGISTRY)}"
        )

    # ── Load data ──────────────────────────────────────────────────────
    data_path = cfg["data"]["raw_path"]
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    df = df.drop(columns=["Time"])

    # ── Preprocess ────────────────────────────────────────────────────
    preprocessor = FraudPreprocessor(
        use_smote=cfg["data"]["use_smote"],
        random_state=cfg["data"]["random_state"],
    )
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        df, test_size=cfg["data"]["test_size"]
    )

    # ── Train ─────────────────────────────────────────────────────────
    model = MODEL_REGISTRY[model_type]
    logger.info("Training %s ...", model_type)
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test, threshold=cfg["model"]["threshold"])

    # ── Save artifacts ────────────────────────────────────────────────
    save_dir = cfg["model"]["save_path"]
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_type}_model.joblib")
    preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")

    joblib.dump(model, model_path)
    preprocessor.save(preprocessor_path)

    logger.info("Model saved to %s", model_path)
    logger.info("Metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
