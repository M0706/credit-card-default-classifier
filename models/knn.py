"""K-Nearest Neighbours, implemented from scratch.

There is no training phase: `fit` simply stores the data. Prediction finds the
k nearest training points by Euclidean distance and lets them vote, weighted by
1/distance so that closer neighbours count more:

    d(x, x') = sqrt( Σ_j (x_j - x'_j)² )

The squared distance is expanded as
    ||x||² - 2·x·x' + ||x'||²
so the whole distance matrix becomes one matrix multiply instead of a loop over
pairs — the same trick scikit-learn uses.

Rows are processed in chunks because an (n_query × n_train) float64 matrix is
large: 6,000 × 24,000 would be 1.15 GB in one go.
"""

import numpy as np

from .base import Model


class KNNModel(Model):
    name = "knn"

    def __init__(self, n_neighbors=15, weights="distance", chunk_size=512):
        super().__init__(n_neighbors=n_neighbors, weights=weights)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.chunk_size = chunk_size

    def fit(self, X, y):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y)
        self.n_features_in_ = self._X.shape[1]
        self.classes_ = np.unique(self._y)
        self._X_sq = np.einsum("ij,ij->i", self._X, self._X)
        return self

    def predict_proba(self, X):
        X = self._validate_X(X)
        n_query = X.shape[0]
        k = min(self.n_neighbors, len(self._X))
        positive = self.classes_[-1]

        out = np.empty((n_query, 2))
        x_sq = np.einsum("ij,ij->i", X, X)

        for start in range(0, n_query, self.chunk_size):
            stop = min(start + self.chunk_size, n_query)
            block = X[start:stop]

            d2 = x_sq[start:stop, None] - 2.0 * (block @ self._X.T) + self._X_sq[None, :]
            np.maximum(d2, 0.0, out=d2)  # rounding can produce tiny negatives

            # argpartition is O(n) per row; it finds the k smallest without
            # sorting all 24,000 candidates.
            neighbours = np.argpartition(d2, k - 1, axis=1)[:, :k]
            dist = np.sqrt(np.take_along_axis(d2, neighbours, axis=1))
            labels = self._y[neighbours]

            if self.weights == "distance":
                # An exact duplicate (distance 0) must dominate the vote;
                # 1/0 would be infinite, so give it an overwhelming weight.
                w = np.where(dist <= 1e-12, 1e12, 1.0 / np.maximum(dist, 1e-12))
            else:
                w = np.ones_like(dist)

            score = (w * (labels == positive)).sum(axis=1) / w.sum(axis=1)
            out[start:stop, 0] = 1.0 - score
            out[start:stop, 1] = score

        return out
