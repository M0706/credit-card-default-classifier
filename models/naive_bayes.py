"""Gaussian Naive Bayes, implemented from scratch.

Bayes' theorem:

    P(y | x) ∝ P(y) · P(x | y)

with the "naive" assumption that the features are conditionally independent
given the class, so P(x | y) factorises into a product over features. Each
factor is modelled as a Gaussian fitted by maximum likelihood (the sample mean
and variance of that feature within the class):

    P(x_i | y) = 1/sqrt(2πσ²_iy) · exp(-(x_i - μ_iy)² / (2σ²_iy))

Everything is computed in log space: multiplying hundreds of small
probabilities underflows to zero in float64, while summing logs does not.

Fitting is just two means and two variances per feature — which is why this
model trains in milliseconds and stores a ~1 KB artefact.
"""

import numpy as np

from .base import Model


class GaussianNBModel(Model):
    name = "naive_bayes"

    def __init__(self, var_smoothing=1e-9):
        super().__init__(var_smoothing=var_smoothing)
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_])

        # A feature that is constant within a class gets variance 0, which would
        # divide by zero below. sklearn's fix: add a tiny fraction of the
        # largest observed variance across all features.
        epsilon = self.var_smoothing * X.var(axis=0).max()
        self.var_ = self.var_ + epsilon

        self.class_prior_ = np.array([(y == c).mean() for c in self.classes_])
        return self

    def _joint_log_likelihood(self, X):
        """log P(y) + Σ_i log P(x_i | y) for every class."""
        jll = np.empty((X.shape[0], len(self.classes_)))
        for k, c in enumerate(self.classes_):
            # log of the Gaussian normaliser: -½ Σ log(2πσ²)
            normaliser = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[k]))
            # exponent term: -½ Σ (x_i - μ)² / σ²
            exponent = -0.5 * np.sum(
                (X - self.theta_[k]) ** 2 / self.var_[k], axis=1
            )
            jll[:, k] = np.log(self.class_prior_[k]) + normaliser + exponent
        return jll

    def predict_proba(self, X):
        X = self._validate_X(X)
        jll = self._joint_log_likelihood(X)
        # subtract the max before exponentiating, otherwise exp() underflows
        jll = jll - jll.max(axis=1, keepdims=True)
        proba = np.exp(jll)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X, threshold=0.5):
        X = self._validate_X(X)
        return self.classes_[self._joint_log_likelihood(X).argmax(axis=1)]
