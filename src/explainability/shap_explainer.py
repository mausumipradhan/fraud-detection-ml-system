"""
explainability/shap_explainer.py
SHAP-based model explanations for individual predictions and global importance.
"""

import numpy as np
import shap
import matplotlib.pyplot as plt
import logging
from typing import Any

logger = logging.getLogger(__name__)


class FraudExplainer:
    """
    Wraps SHAP TreeExplainer for XGBoost / Random Forest models.
    Falls back to KernelExplainer for linear models.
    """

    def __init__(self, model: Any, feature_names: list[str] | None = None):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None

    def _get_explainer(self, X_background: np.ndarray | None = None):
        if self._explainer is not None:
            return self._explainer

        try:
            self._explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer")
        except Exception:
            if X_background is None:
                raise ValueError("X_background required for KernelExplainer")
            self._explainer = shap.KernelExplainer(
                self.model.predict_proba, shap.sample(X_background, 100)
            )
            logger.info("Using KernelExplainer")
        return self._explainer

    def explain_instance(
        self, x: np.ndarray, X_background: np.ndarray | None = None, top_n: int = 10
    ) -> dict:
        """
        Explain a single prediction.

        Returns:
            dict with top feature contributions and base value.
        """
        explainer = self._get_explainer(X_background)
        shap_values = explainer.shap_values(x.reshape(1, -1))

        # For binary classifiers, take fraud class values
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        names = self.feature_names or [f"V{i+1}" for i in range(len(shap_vals))]
        pairs = sorted(zip(names, shap_vals), key=lambda p: abs(p[1]), reverse=True)

        top_features = [
            {"feature": name, "impact": round(float(val), 4)}
            for name, val in pairs[:top_n]
        ]

        return {
            "top_features": top_features,
            "base_value": round(float(explainer.expected_value if not isinstance(
                explainer.expected_value, list
            ) else explainer.expected_value[1]), 4),
        }

    def plot_summary(
        self, X: np.ndarray, max_display: int = 20, save_path: str | None = None
    ):
        """Global feature importance summary plot."""
        explainer = self._get_explainer(X)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        return plt.gcf()

    def plot_waterfall(
        self, x: np.ndarray, X_background: np.ndarray | None = None, save_path: str | None = None
    ):
        """Waterfall plot for a single prediction."""
        explainer = self._get_explainer(X_background)
        sv = explainer(x.reshape(1, -1))
        shap.plots.waterfall(sv[0], show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        return plt.gcf()
