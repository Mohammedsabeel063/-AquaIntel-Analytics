"""
AquaIntel Analytics - Model Utilities
Contains shared model classes and utilities
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SoftVotingHybrid(BaseEstimator, ClassifierMixin):
    """Soft voting ensemble combining RF and XGB predictions."""
    def __init__(self, rf_model=None, xgb_model=None):
        self.rf_model = rf_model
        self.xgb_model = xgb_model

    def fit(self, X, y):
        """Fit both models (no-op if models already provided)."""
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Both rf_model and xgb_model must be provided")
        return self

    def predict_proba(self, X):
        """Average probabilities from both models."""
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        return (rf_proba + xgb_proba) / 2.0

    def predict(self, X):
        """Predict based on averaged probabilities."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)