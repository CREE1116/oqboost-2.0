"""Functional tests for OQBoost — core, features, and edge cases."""
import numpy as np
import pickle
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score

from oqboost import OQBoostClassifier, OQBoostRegressor


@pytest.fixture
def clf_data():
    return make_classification(400, 10, n_informative=6, random_state=0)


@pytest.fixture
def reg_data():
    return make_regression(400, 8, noise=8.0, random_state=0)


# ── core ────────────────────────────────────────────────────────────────────
def test_classifier_basic(clf_data):
    X, y = clf_data
    m = OQBoostClassifier(n_estimators=40, random_state=0).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(y), 2)
    assert np.allclose(p.sum(1), 1)
    assert set(m.predict(X)) <= set(y)
    assert roc_auc_score(y, p[:, 1]) > 0.9


def test_regressor_basic(reg_data):
    X, y = reg_data
    m = OQBoostRegressor(n_estimators=60, random_state=0).fit(X, y)
    assert m.predict(X).shape == (len(y),)
    assert r2_score(y, m.predict(X)) > 0.8


def test_multiclass_ovr():
    X, y = load_iris(return_X_y=True)
    m = OQBoostClassifier(n_estimators=60, random_state=0).fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (150, 3) and np.allclose(P.sum(1), 1)
    assert accuracy_score(y, m.predict(X)) > 0.95


@pytest.mark.parametrize("Est", [OQBoostClassifier, OQBoostRegressor])
def test_pickle_clone(Est, clf_data):
    X, y = clf_data
    m = Est(n_estimators=20, random_state=0).fit(X, y.astype(float))
    pred = m.predict(X)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.array_equal(m2.predict(X), pred)
    clone(m)  # unfitted clone must not raise


# ── inputs ──────────────────────────────────────────────────────────────────
def test_nan_native(clf_data):
    X, y = clf_data
    X = X.copy(); X[::9, 2] = np.nan
    m = OQBoostClassifier(n_estimators=20, random_state=0).fit(X, y)
    assert np.isfinite(m.predict_proba(X)).all()


def test_sparse_equals_dense(clf_data):
    X, y = clf_data
    d = OQBoostClassifier(n_estimators=20, random_state=0).fit(X, y).predict_proba(X)
    s = OQBoostClassifier(n_estimators=20, random_state=0).fit(csr_matrix(X), y).predict_proba(csr_matrix(X))
    assert np.allclose(d, s)


def test_pandas_feature_names(clf_data):
    pd = pytest.importorskip("pandas")
    X, y = clf_data
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    m = OQBoostClassifier(n_estimators=20, random_state=0).fit(df, y)
    assert list(m.feature_names_in_) == list(df.columns)


# ── weighting ───────────────────────────────────────────────────────────────
def test_sample_weight_identity(clf_data):
    X, y = clf_data
    a = OQBoostClassifier(random_state=0).fit(X, y).predict_proba(X)
    b = OQBoostClassifier(random_state=0).fit(X, y, sample_weight=np.ones(len(y))).predict_proba(X)
    assert np.allclose(a, b)


def test_sample_weight_extreme(clf_data):
    X, y = clf_data
    w = np.ones(len(y)); w[0] = 5000
    yf = y.copy(); yf[0] = 1 - y[0]
    p = OQBoostClassifier(n_estimators=80, subsample=1.0, random_state=0).fit(
        X, yf, sample_weight=w).predict_proba(X[:1])[0, 1]
    assert (p > 0.5) == bool(yf[0])


def test_class_weight_runs(clf_data):
    X, y = clf_data
    for cw in ["balanced", {0: 1, 1: 3}]:
        OQBoostClassifier(n_estimators=20, class_weight=cw, random_state=0).fit(X, y)


# ── early stopping ──────────────────────────────────────────────────────────
def test_early_stopping_truncates(clf_data):
    X, y = clf_data
    m = OQBoostClassifier(n_estimators=1000, n_iter_no_change=10,
                          validation_fraction=0.2, random_state=0).fit(X, y)
    assert m._booster.n_trees() < 1000
    assert m.best_iteration_ >= 0


def test_no_early_stopping_full(clf_data):
    X, y = clf_data
    m = OQBoostClassifier(n_estimators=50, random_state=0).fit(X, y)
    assert m._booster.n_trees() == 50


# ── features ────────────────────────────────────────────────────────────────
def test_warm_start_adds_trees(reg_data):
    X, y = reg_data
    m = OQBoostRegressor(n_estimators=30, warm_start=True, random_state=0).fit(X, y)
    m.set_params(n_estimators=60).fit(X, y)
    assert m._booster.n_trees() == 60


def test_monotonic(reg_data):
    rng = np.random.default_rng(0)
    n, d = 2000, 4
    X = rng.uniform(-2, 2, (n, d))
    y = X[:, 0] - 0.5 * np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.2, n)
    m = OQBoostRegressor(monotone_constraints={0: 1}, random_state=0).fit(X, y)
    xs = np.linspace(-2, 2, 40)
    worst = 0.0
    for _ in range(200):
        XX = np.tile(rng.uniform(-2, 2, d), (40, 1)); XX[:, 0] = xs
        worst = min(worst, np.diff(m.predict(XX)).min())
    assert worst >= -1e-6  # monotone non-decreasing in feature 0


def test_robust_loss(reg_data):
    X, y = reg_data
    OQBoostRegressor(loss="huber", clip=True, random_state=0).fit(X, y)
    OQBoostRegressor(loss="quantile", alpha=0.9, random_state=0).fit(X, y)


def test_explain_additive(reg_data):
    X, y = reg_data
    m = OQBoostRegressor(n_estimators=60, random_state=0).fit(X, y)
    phi = m.explain(X[:20])
    assert phi.shape == (20, X.shape[1])
    # additive: sum(phi) == prediction - base (init = mean(y))
    assert np.allclose(phi.sum(1), m.predict(X[:20]) - y.mean(), atol=1e-6)


def test_explain_multiclass_shape():
    X, y = load_iris(return_X_y=True)
    m = OQBoostClassifier(n_estimators=40, random_state=0).fit(X, y)
    assert m.explain(X[:5]).shape == (5, 3, 4)


@pytest.mark.parametrize("cf", [[0], [True, False], np.array([0])])
def test_categorical_features_accepts_formats(cf):
    # index list / bool mask / array all resolve to the same column
    rng = np.random.default_rng(0); n = 600
    X = np.column_stack([rng.integers(0, 20, n).astype(float), rng.normal(size=n)])
    y = (rng.random(n) < 0.5).astype(int)
    m = OQBoostClassifier(n_estimators=20, categorical_features=cf, random_state=0).fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (n, 2) and np.isfinite(p).all()


def test_categorical_target_encoding_gain():
    # high-cardinality (100 > max_bins) category as signal: target encoding (the
    # categorical_features path) should recover the signal a linear oblique split
    # can't read from raw codes. Big, reproducible gain.
    rng = np.random.default_rng(0); n, card = 8000, 100
    offs = rng.normal(0, 2.5, card); code = rng.integers(0, card, n); c1 = rng.normal(size=n)
    logit = offs[code] + 0.4 * c1 + 0.5 * (offs[code] > 0) * c1
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = np.column_stack([code.astype(float), c1])
    Xtr, Xte, ytr, yte = X[:6000], X[6000:], y[:6000], y[6000:]
    a_raw = roc_auc_score(yte, OQBoostClassifier(n_estimators=200, random_state=0)
                          .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    a_te = roc_auc_score(yte, OQBoostClassifier(n_estimators=200, categorical_features=[0],
                         random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    assert a_te > a_raw + 0.08


def test_categorical_predict_no_leak():
    # cross-fitted encoding -> held-out score must stay realistic (not ~1.0)
    rng = np.random.default_rng(1); n, card = 4000, 80
    offs = rng.normal(0, 2, card); code = rng.integers(0, card, n)
    X = np.column_stack([code.astype(float), rng.normal(size=n)])
    y = (rng.random(n) < 1 / (1 + np.exp(-offs[code]))).astype(int)
    Xtr, Xte, ytr, yte = X[:3000], X[3000:], y[:3000], y[3000:]
    m = OQBoostClassifier(n_estimators=150, categorical_features=[0], random_state=0).fit(Xtr, ytr)
    train_auc = roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1])
    test_auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    assert test_auc > 0.6 and train_auc - test_auc < 0.15  # no leak blowup


# ── edge cases ──────────────────────────────────────────────────────────────
def test_single_class_raises(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError):
        OQBoostClassifier(n_estimators=10).fit(X, np.zeros(len(y)))


def test_continuous_target_raises(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError):
        OQBoostClassifier(n_estimators=10).fit(X, X[:, 0])


def test_all_zero_weight_raises(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError):
        OQBoostClassifier(n_estimators=10).fit(X, y, sample_weight=np.zeros(len(y)))


def test_wrong_n_features_raises(clf_data):
    X, y = clf_data
    m = OQBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X[:, :3])


def test_y_none_raises(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError):
        OQBoostClassifier(n_estimators=10).fit(X, None)
