"""Support Vector Machine with an RBF kernel, implemented from scratch (SMO).

The SVM finds the decision boundary with the widest "margin" between the two
classes, allowing some training points to violate it. Written as a
minimisation problem (t_i ∈ {-1, +1} is the signed label):

    min_α  ½ Σ_i Σ_j α_i α_j t_i t_j K(x_i, x_j) - Σ_i α_i
    s.t.   0 ≤ α_i ≤ C        and       Σ_i α_i t_i = 0

The RBF kernel measures similarity between two points and vanishes as they get
further apart:

    K(x, x') = exp(-γ ‖x - x'‖²)

Only points with α_i > 0 matter — the "support vectors". Everything else could
be deleted from the training set and the boundary would be unchanged, which is
the whole point of an SVM.

How it is solved: SMO
---------------------
Solving that quadratic program for all α at once is expensive. Sequential
Minimal Optimisation instead takes two multipliers at a time — a pair is the
smallest group that can be adjusted while still satisfying Σ α_i t_i = 0 — and
solves that 2-variable sub-problem exactly, in closed form. Sweeping over all
pairs until no multiplier changes by more than `tol` converges to the global
optimum, because the objective is convex.

The kernel matrix is precomputed once. That costs O(n²) memory, so for large
training sets we fit on a stratified subsample (`max_samples`). That is the one
honest compromise in this implementation, and validate_models.py reports the
cost: the subsampled model is compared against scikit-learn's on the same
subsample, so the *implementation* is verified even though the *model* sees
less data than the other four.
"""

import numpy as np

from .base import Model


class SVMModel(Model):
    name = "svm"

    def __init__(
        self,
        C=1.0,
        gamma="scale",
        tol=1e-3,
        max_passes=5,
        max_samples=4000,
        random_state=42,
    ):
        super().__init__(
            C=C,
            gamma=gamma,
            tol=tol,
            max_passes=max_passes,
            max_samples=max_samples,
            random_state=random_state,
        )
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_samples = max_samples
        self.random_state = random_state

    # ------------------------------------------------------------------ fit

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        X, y = self._subsample(X, y)
        t = np.where(y == self.classes_[-1], 1.0, -1.0)
        n = len(t)

        self._gamma = self._resolve_gamma(X)
        kernel = self._rbf(X, X)

        alphas = np.zeros(n)
        b = 0.0
        # f[i] is the current decision value for training point i; keeping it
        # updated incrementally avoids recomputing the sum for every check.
        f = np.zeros(n)
        rng = np.random.default_rng(self.random_state)

        passes = 0
        sweeps = 0
        while passes < self.max_passes:
            sweeps += 1
            changed = 0
            for i in range(n):
                error_i = f[i] - t[i]
                violated = (t[i] * error_i < -self.tol and alphas[i] < self.C) or (
                    t[i] * error_i > self.tol and alphas[i] > 0
                )
                if not violated:
                    continue

                # Pick the second multiplier at random among the rest.
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1
                error_j = f[j] - t[j]

                alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                if t[i] != t[j]:
                    low = max(0.0, alpha_j_old - alpha_i_old)
                    high = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    low = max(0.0, alpha_i_old + alpha_j_old - self.C)
                    high = min(self.C, alpha_i_old + alpha_j_old)

                if high - low < 1e-12:
                    continue

                # Second derivative of the objective along the pair.
                eta = 2.0 * kernel[i, j] - kernel[i, i] - kernel[j, j]
                if eta >= -1e-12:
                    continue

                alphas[j] = alpha_j_old - t[j] * (error_i - error_j) / eta
                alphas[j] = min(high, max(low, alphas[j]))
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    alphas[j] = alpha_j_old
                    continue

                alphas[i] = alpha_i_old + t[i] * t[j] * (alpha_j_old - alphas[j])

                b1 = (
                    b
                    - error_i
                    - t[i] * (alphas[i] - alpha_i_old) * kernel[i, i]
                    - t[j] * (alphas[j] - alpha_j_old) * kernel[i, j]
                )
                b2 = (
                    b
                    - error_j
                    - t[i] * (alphas[i] - alpha_i_old) * kernel[i, j]
                    - t[j] * (alphas[j] - alpha_j_old) * kernel[j, j]
                )
                if 0 < alphas[i] < self.C:
                    new_b = b1
                elif 0 < alphas[j] < self.C:
                    new_b = b2
                else:
                    new_b = (b1 + b2) / 2.0

                delta_i = (alphas[i] - alpha_i_old) * t[i]
                delta_j = (alphas[j] - alpha_j_old) * t[j]
                # f holds the full decision value, so it must absorb the bias
                # change too — not just the multiplier changes.
                f += delta_i * kernel[i] + delta_j * kernel[j] + (new_b - b)
                b = new_b
                changed += 1

            if changed == 0:
                passes += 1
            else:
                passes = 0

        support = alphas > 1e-8
        self.support_vectors_ = X[support]
        self.dual_coef_ = (alphas * t)[support]
        self.intercept_ = b
        self.n_support_ = int(support.sum())
        self.n_iter_ = sweeps
        return self

    def _subsample(self, X, y):
        if self.max_samples is None or len(y) <= self.max_samples:
            return X, y
        rng = np.random.default_rng(self.random_state)
        # Stratified: keep the class balance of the full training set.
        keep = []
        for label in np.unique(y):
            rows = np.flatnonzero(y == label)
            share = int(round(self.max_samples * len(rows) / len(y)))
            keep.append(rng.choice(rows, size=min(share, len(rows)), replace=False))
        index = np.sort(np.concatenate(keep))
        return X[index], y[index]

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            variance = X.var()
            return 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        return float(self.gamma)

    # ------------------------------------------------------------- predicting

    def decision_function(self, X):
        X = self._validate_X(X)
        if getattr(self, "support_vectors_", None) is None or len(self.dual_coef_) == 0:
            return np.zeros(X.shape[0])
        out = np.empty(X.shape[0])
        chunk = 256
        for start in range(0, X.shape[0], chunk):
            stop = min(start + chunk, X.shape[0])
            out[start:stop] = self._rbf(X[start:stop], self.support_vectors_) @ self.dual_coef_
        return out + self.intercept_

    def predict_proba(self, X):
        # sigmoid of the decision value. Monotone in the decision value, so the
        # ranking (and therefore AUC) is exact; the probabilities themselves are
        # not calibrated. Documented in the README.
        p = self._sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold=0.5):
        X = self._validate_X(X)
        return np.where(self.decision_function(X) >= 0.0, self.classes_[-1], self.classes_[0])

    # ---------------------------------------------------------------- kernel

    def _rbf(self, A, B):
        a_sq = np.einsum("ij,ij->i", A, A)
        b_sq = np.einsum("ij,ij->i", B, B)
        sq = a_sq[:, None] - 2.0 * (A @ B.T) + b_sq[None, :]
        np.maximum(sq, 0.0, out=sq)
        return np.exp(-self._gamma * sq)
