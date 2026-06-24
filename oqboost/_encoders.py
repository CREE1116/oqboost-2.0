"""Target encoding for categorical columns (empirical-Bayes "auto" smoothing).

The numeric kernel — fold assignment, cross-fitted level statistics, and
smoothing — lives in C++ (`oqboost_core.make_folds` / `te_fit_transform` /
`te_transform`); this module only maps raw values to contiguous codes and
replaces the categorical columns in-place. Encoding is fit once on the training
data; the fitted level maps are reused at predict. No sklearn dependency.
"""
import numpy as np

from . import oqboost_core as _core


def _codes(values, levels):
    """Map raw values to 0..K-1 against sorted unique `levels`; unseen -> -1."""
    idx = np.searchsorted(levels, values)
    idx = np.clip(idx, 0, len(levels) - 1)
    match = levels[idx] == values
    return np.where(match, idx, -1).astype(np.int64)


def fit_transform(X, cat_idx, y, classification, n_folds=5, seed=0):
    """Cross-fitted target encoding of `cat_idx` columns.

    Returns (X_encoded, encoders); `encoders[i] = (levels, full_map, gmean)` for
    cat_idx[i], to be replayed by `transform`.
    """
    X = np.array(X, dtype=float, order="C")
    yf = np.ascontiguousarray(np.asarray(y, dtype=float))
    n = len(yf)
    nf = max(2, min(n_folds, n))
    folds = _core.make_folds(yf, nf, int(seed), bool(classification))
    encoders = []
    for j in cat_idx:
        levels = np.unique(X[:, j])
        codes = _codes(X[:, j], levels)
        enc, full_map, gmean = _core.te_fit_transform(codes, yf, folds, len(levels), nf)
        X[:, j] = enc
        encoders.append((levels, np.asarray(full_map, dtype=float), float(gmean)))
    return X, encoders


def transform(X, cat_idx, encoders):
    """Apply fitted encoders to a copy of X (categorical columns replaced)."""
    X = np.array(X, dtype=float, order="C")
    for j, (levels, full_map, gmean) in zip(cat_idx, encoders):
        codes = _codes(X[:, j], levels)
        X[:, j] = _core.te_transform(codes, full_map, gmean)
    return X
