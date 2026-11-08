"""Common interface shared by all five classifiers.

Every model exposes the same three methods, which is what lets
train_models.py and app.py treat them interchangeably:

    fit(X, y)              learn from labelled rows
    predict_proba(X)       -> (n_samples, 2) array of class probabilities
    predict(X, threshold)  -> (n_samples,) array of 0/1 labels

The contract is deliberately the same shape as scikit-learn's, which makes
validate_models.py able to score both implementations with identical code.
"""

from abc import ABC, abstractmethod

import joblib
import numpy as np


class Model(ABC):
    """Base class for the hand-written classifiers."""

    name = "model"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def fit(self, X, y):
        """Learn parameters. Must set self.n_features_in_ and self.classes_."""

    @abstractmethod
    def predict_proba(self, X):
        """Return an (n_samples, 2) array: column 0 = P(class 0), column 1 = P(class 1)."""

    def predict(self, X, threshold=0.5):
        """Label as 1 where P(class 1) reaches the threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        return X

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
