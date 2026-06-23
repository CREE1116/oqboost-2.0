"""OQBoost 2.0 — gradient-boosted 2D-oblique trees (C++ backend, scikit-learn API)"""
from ._sklearn import OQBoostClassifier, OQBoostRegressor

__version__ = "2.0.0"
__all__ = ["OQBoostClassifier", "OQBoostRegressor"]
