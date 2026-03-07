"""
preprocessing/pipeline.py
Feature engineering, scaling, and SMOTE oversampling pipeline.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """
    End-to-end preprocessing pipeline for credit card fraud detection.

    Steps:
        1. Drop duplicates
        2. Scale 'Amount' and 'Time' features
        3. Apply SMOTE to balance classes (training only)
    """

    def __init__(self, use_smote: bool = True, random_state: int = 42):
        self.use_smote = use_smote
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = "Class",
        test_size: float = 0.2,
    ) -> tuple:
        """
        Full preprocessing on raw DataFrame.

        Returns:
            X_train, X_test, y_train, y_test (all numpy arrays)
        """
        df = self._clean(df)
        X, y = self._split_features(df, target_col)
        self.feature_names = list(X.columns)

        X_scaled = self._scale_fit(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y.values,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        if self.use_smote:
            X_train, y_train = self._apply_smote(X_train, y_train)

        self._fitted = True
        logger.info(
            "Preprocessing complete | train=%d test=%d | fraud_ratio_train=%.4f",
            len(X_train),
            len(X_test),
            y_train.mean(),
        )
        return X_train, X_test, y_train, y_test

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform new data using the fitted scaler."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]
        return self.scaler.transform(X)

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FraudPreprocessor":
        obj = joblib.load(path)
        logger.info("Preprocessor loaded from %s", path)
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        logger.info("Dropped %d duplicate rows", before - len(df))
        return df

    def _split_features(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        logger.info(
            "Class distribution — 0: %d  1: %d  (fraud %.4f%%)",
            (y == 0).sum(),
            (y == 1).sum(),
            y.mean() * 100,
        )
        return X, y

    def _scale_fit(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sm = SMOTE(random_state=self.random_state)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info("SMOTE applied | before: %d  after: %d", len(X), len(X_res))
        return X_res, y_res
