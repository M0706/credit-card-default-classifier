"""Logistic Regression, implemented from scratch with Newton's method (IRLS).

The model is a linear function of the features pushed through a sigmoid, so the
output is a probability:

    z = w·x + b        p = σ(z) = 1 / (1 + e^(-z))
    log(p / (1-p)) = w·x + b

Training minimises L2-penalised log loss (the same objective scikit-learn uses
with penalty="l2"):

    J(w, b) = ½‖w‖² + C · Σ_i [ -y_i log p_i - (1-y_i) log(1-p_i) ]

`C` is scikit-learn's inverse-regularisation convention: larger C = weaker
penalty. The intercept is not penalised.

Why Newton rather than gradient descent
---------------------------------------
Here there are only 23 features, so the Hessian is 24×24 and cheap to invert
exactly. Newton's method solves

    β ← β - H⁻¹ g

and converges in a handful of iterations regardless of feature scaling, whereas
plain gradient descent needs a hand-tuned learning rate and hundreds of steps.
This is the same family of solver scikit-learn's lbfgs uses, just the classic
Newton–Raphson version (IRLS — iteratively reweighted least squares).

    g      = ∂J/∂β
    H      = ∂²J/∂β²  =  C · Xᵀ W X + (penalty on the weight block only)
    W_ii   = p_i (1 - p_i)
"""

import numpy as np

from .base import Model


class LogisticRegressionModel(Model):
    name = "logistic_regression"

    def __init__(self, C=1.0, max_iter=100, tol=1e-8):
        super().__init__(C=C, max_iter=max_iter, tol=tol)
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        self.classes_ = np.array([0, 1])

        beta = np.zeros(d + 1)  # beta[0] = intercept
        for iteration in range(self.max_iter):
            p = self._sigmoid(X @ beta[1:] + beta[0])
            residual = p - y

            grad = np.empty(d + 1)
            grad[0] = self.C * residual.sum()
            grad[1:] = beta[1:] + self.C * (X.T @ residual)

            w = np.clip(p * (1.0 - p), 1e-12, None)
            hessian = np.empty((d + 1, d + 1))
            hessian[0, 0] = self.C * w.sum()
            hessian[0, 1:] = self.C * (w @ X)
            hessian[1:, 0] = hessian[0, 1:]
            hessian[1:, 1:] = np.eye(d) + self.C * (X.T @ (X * w[:, None]))

            step = np.linalg.solve(hessian + 1e-10 * np.eye(d + 1), grad)
            beta -= step
            if np.max(np.abs(step)) < self.tol:
                break

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]
        self.n_iter_ = iteration + 1
        return self

    def predict_proba(self, X):
        X = self._validate_X(X)
        p = self._sigmoid(X @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p, p])

    def decision_function(self, X):
        X = self._validate_X(X)
        return X @ self.coef_ + self.intercept_

    def predict(self, X, threshold=0.5):
        X = self._validate_X(X)
        return (self.decision_function(X) >= 0).astype(int)
