"""
monitoring/drift_detector.py
Statistical data drift detection using Population Stability Index (PSI)
and Kolmogorov-Smirnov test.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    timestamp: str
    feature_drifts: (
        dict  # {feature: {"psi": float, "ks_pvalue": float, "drifted": bool}}
    )
    overall_drift: bool
    drifted_features: list[str]
    recommendation: str


class DriftDetector:
    """
    Detects concept and data drift by comparing incoming transactions
    against a reference distribution (training data).

    Checks:
    - Population Stability Index (PSI): psi > 0.2 → drift
    - KS test p-value: p < 0.05 → drift
    """

    PSI_THRESHOLD = 0.2
    KS_PVALUE_THRESHOLD = 0.05

    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: list[str] | None = None,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
    ):
        self.reference_data = reference_data
        self.feature_names = feature_names or [
            f"V{i+1}" for i in range(reference_data.shape[1])
        ]
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self._buffer: deque = deque(maxlen=window_size)
        self._alerts: list[DriftReport] = []

    # ------------------------------------------------------------------
    def add_transaction(self, features: np.ndarray) -> None:
        """Add a transaction to the monitoring window."""
        self._buffer.append(features)

    def check_drift(self) -> DriftReport | None:
        """
        Run drift checks on the current window.
        Returns a DriftReport, or None if window is not full yet.
        """
        if len(self._buffer) < self.window_size:
            logger.debug("Buffer not full (%d/%d)", len(self._buffer), self.window_size)
            return None

        current_data = np.array(self._buffer)
        feature_drifts = {}
        drifted = []

        for i, name in enumerate(self.feature_names):
            ref_col = self.reference_data[:, i]
            cur_col = current_data[:, i]

            psi = self._compute_psi(ref_col, cur_col)
            ks_stat, ks_pvalue = stats.ks_2samp(ref_col, cur_col)

            is_drifted = (
                psi > self.PSI_THRESHOLD or ks_pvalue < self.KS_PVALUE_THRESHOLD
            )

            feature_drifts[name] = {
                "psi": round(psi, 4),
                "ks_pvalue": round(ks_pvalue, 4),
                "drifted": is_drifted,
            }
            if is_drifted:
                drifted.append(name)

        overall = len(drifted) / len(self.feature_names) > self.drift_threshold

        rec = "Model retraining recommended." if overall else "No action required."

        report = DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            feature_drifts=feature_drifts,
            overall_drift=overall,
            drifted_features=drifted,
            recommendation=rec,
        )

        if overall:
            logger.warning(
                "DRIFT DETECTED — %d features drifted: %s", len(drifted), drifted
            )
        else:
            logger.info("No significant drift detected.")

        self._alerts.append(report)
        return report

    # ------------------------------------------------------------------
    def _compute_psi(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """Population Stability Index."""
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)

        ref_counts = np.histogram(reference, bins=breakpoints)[0] + 1e-6
        cur_counts = np.histogram(current, bins=breakpoints)[0] + 1e-6

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)
