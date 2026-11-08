"""CART decision tree, implemented from scratch.

A tree is grown greedily: at every node we search for the single (feature,
threshold) pair that best separates the classes, split the data on it, and
recurse into each half. "Best" means the largest drop in Gini impurity.

For a binary problem let p be the fraction of positives in a node. Then

    Gini(p) = 1 - p² - (1-p)² = 2p(1-p)

which is 0 when the node is pure and 0.5 when it is 50/50. A candidate split
divides the node into left (fraction n_L/n, purity p_L) and right (n_R/n, p_R),
and the impurity drop is

    Δ = Gini(p) - [ (n_L/n)·Gini(p_L) + (n_R/n)·Gini(p_R) ]

The split with the largest Δ wins. Repeating this recursively is what builds
the tree; the stopping rules (max_depth, min_samples_leaf) stop it from growing
until every leaf is a single training point — which would memorise the data.

One deliberate difference from scikit-learn: candidate thresholds here are
quantiles of each feature rather than "every midpoint between adjacent sorted
values". With ~19k rows and 23 features the exact search examines millions of
candidates per node, while 32 quantile thresholds capture essentially the same
split. The validation harness measures how much accuracy that approximation
actually costs (in practice: almost none).
"""

import numpy as np

from .base import Model


class DecisionTreeModel(Model):
    name = "decision_tree"

    def __init__(self, max_depth=4, min_samples_leaf=50, max_thresholds=32):
        super().__init__(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_thresholds=max_thresholds,
        )
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._positive = self.classes_[-1]
        self.root_ = self._grow(X, y, depth=0)
        self.n_leaves_ = self._count_leaves(self.root_)
        self.feature_importances_ = self._importances()
        return self

    # ---------------------------------------------------------------- growing

    def _grow(self, X, y, depth):
        positive = (y == self._positive)
        p = positive.mean()
        node = {"prob": float(p), "n": len(y), "impurity": 2.0 * p * (1.0 - p)}

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < 2 * self.min_samples_leaf
            or p == 0.0
            or p == 1.0
        ):
            return node

        feature, threshold, gain = self._best_split(X, positive)
        if feature is None or gain <= 1e-12:
            return node

        left_mask = X[:, feature] <= threshold
        node["feature"] = feature
        node["threshold"] = threshold
        node["left"] = self._grow(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X, positive):
        n = len(positive)
        pos = positive.astype(float)
        parent = 2.0 * pos.mean() * (1.0 - pos.mean())
        best_feature, best_threshold, best_gain = None, None, 0.0

        for feature in range(X.shape[1]):
            column = X[:, feature]
            unique = np.unique(column)
            if unique.size < 2:
                continue

            if unique.size > self.max_thresholds:
                qs = np.linspace(0.0, 1.0, self.max_thresholds + 2)[1:-1]
                candidates = np.unique(np.quantile(unique, qs))
            else:
                candidates = (unique[:-1] + unique[1:]) / 2.0
            if candidates.size == 0:
                continue

            # Sorting once per feature lets every candidate be evaluated with
            # a prefix sum instead of re-partitioning the data each time.
            order = np.argsort(column, kind="mergesort")
            sorted_col = column[order]
            cumulative_pos = np.cumsum(pos[order])

            idx = np.searchsorted(sorted_col, candidates, side="right")
            keep = (idx > 0) & (idx < n)
            if not keep.any():
                continue
            idx = idx[keep]
            candidates = candidates[keep]

            left_n = idx.astype(float)
            right_n = n - left_n
            left_pos = cumulative_pos[idx - 1]
            right_pos = cumulative_pos[-1] - left_pos

            left_p, right_p = left_pos / left_n, right_pos / right_n
            impurity = (
                left_n * 2.0 * left_p * (1.0 - left_p)
                + right_n * 2.0 * right_p * (1.0 - right_p)
            ) / n

            gains = parent - impurity
            gains[(left_n < self.min_samples_leaf) | (right_n < self.min_samples_leaf)] = -np.inf

            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_feature = feature
                best_threshold = float(candidates[j])
                best_gain = float(gains[j])

        return best_feature, best_threshold, best_gain

    # -------------------------------------------------------------- predicting

    def predict_proba(self, X):
        X = self._validate_X(X)
        out = np.empty(X.shape[0])
        self._apply(self.root_, X, np.arange(X.shape[0]), out)
        return np.column_stack([1.0 - out, out])

    def _apply(self, node, X, rows, out):
        if "feature" not in node:
            out[rows] = node["prob"]
            return
        go_left = X[rows, node["feature"]] <= node["threshold"]
        self._apply(node["left"], X, rows[go_left], out)
        self._apply(node["right"], X, rows[~go_left], out)

    # ---------------------------------------------------------------- helpers

    def _count_leaves(self, node):
        if "feature" not in node:
            return 1
        return self._count_leaves(node["left"]) + self._count_leaves(node["right"])

    def _importances(self):
        """Total impurity decrease attributable to each feature, normalised."""
        totals = np.zeros(self.n_features_in_)

        def walk(node):
            if "feature" not in node:
                return
            left, right = node["left"], node["right"]
            child = (left["n"] * left["impurity"] + right["n"] * right["impurity"]) / node["n"]
            totals[node["feature"]] += node["n"] * (node["impurity"] - child)
            walk(left)
            walk(right)

        walk(self.root_)
        total = totals.sum()
        return totals / total if total > 0 else totals
