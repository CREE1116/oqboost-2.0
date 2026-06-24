"""scikit-learn estimator compliance via check_estimator.

The two sample-weight-equivalence checks are expected to fail: exact equivalence
between weighted fitting and repeating rows (to rtol 1e-7) is infeasible for a
stochastic, histogram-binned booster (subsample RNG + unweighted bin edges).
"""
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from oqboost import OQBoostClassifier, OQBoostRegressor

_EXPECTED_FAILED = {
    "check_sample_weight_equivalence_on_dense_data":
        "binned/stochastic booster: weighted fit != repeated rows exactly",
    "check_sample_weight_equivalence_on_sparse_data":
        "binned/stochastic booster: weighted fit != repeated rows exactly",
}


@parametrize_with_checks(
    [OQBoostClassifier(n_estimators=20), OQBoostRegressor(n_estimators=20)],
    expected_failed_checks=lambda est: _EXPECTED_FAILED,
)
def test_sklearn_compliance(estimator, check):
    check(estimator)
