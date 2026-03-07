"""
models/evaluate.py
Evaluation utilities: metrics, ROC curve, confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Returns:
        dict with roc_auc, precision, recall, f1, average_precision
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "average_precision": round(average_precision_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "threshold": threshold,
    }

    logger.info("=== Evaluation Results ===")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    logger.info(
        "\n%s", classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    )

    return metrics


def plot_roc_curve(
    model, X_test: np.ndarray, y_test: np.ndarray, save_path: str | None = None
):
    """Plot and optionally save ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()


def plot_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, save_path: str | None = None
):
    """Plot confusion matrix heatmap."""
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return plt.gcf()
