"""Hand-written implementations of the five classifiers used in this assignment.

Each algorithm lives in its own module and subclasses `Model`, so training and
serving code never has to care which one it is holding.
"""

from .base import Model
from .knn import KNNModel
from .logistic import LogisticRegressionModel
from .naive_bayes import GaussianNBModel
from .svm import SVMModel
from .tree import DecisionTreeModel

__all__ = [
    "Model",
    "LogisticRegressionModel",
    "DecisionTreeModel",
    "KNNModel",
    "GaussianNBModel",
    "SVMModel",
]
