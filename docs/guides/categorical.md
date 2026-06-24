# Categorical features

Pass `categorical_features` (column indices or a boolean mask) to **target-encode**
those columns: each level is replaced by a cross-fitted estimate of the target
conditioned on that level.

```python
clf = OQBoostClassifier(categorical_features=[0, 4]).fit(X, y)
# or a boolean mask
clf = OQBoostClassifier(categorical_features=[True, False, False, False, True]).fit(X, y)
```

Encode categories as **integer codes** (`0..K-1`) before fitting (e.g. with
`sklearn.preprocessing.OrdinalEncoder`).

## Why target encoding (and not one-hot / lossless bins)

An oblique split `a·cat + b·cont < t` imposes a **linear order** on the category
codes. Nominal codes have no meaningful order, so a split on raw codes is
useless. Target encoding orders the levels by their effect on the target, which
makes `a·TE(cat)` a meaningful axis **and** lets `a·TE(cat) + b·cont` capture
genuine category×continuous interactions — exactly what the oblique core is good
at. On high-cardinality signal this is a large, reproducible win (synthetic
card-100: +0.16 AUC binary, +0.56 R² regression, +0.24 accuracy multiclass over
raw codes).

This is why OQBoost target-encodes rather than one-hot encoding or lossless
binning: target encoding helps oblique splits *more* than it helps axis-aligned
trees (which can set-partition a category and so need the ordering less).

## How it works

- **Empirical-Bayes "auto" smoothing** (Micci-Barreca): rare levels are shrunk
  toward the global target mean by their count, so low-support levels don't
  overfit. No tuning knob.
- **Cross-fitted** (5-fold) at training time so a row's encoding never uses its
  own target — this prevents leakage. At predict time the full-train level map is
  applied; unseen levels fall back to the global mean.
- The numeric kernel (fold assignment, level statistics, smoothing) is implemented
  in C++; the encoding is fit once and reused at predict.

## Tasks

- **Binary / regression**: one encoded column per categorical, computed against
  the target.
- **Multiclass** (one-vs-rest): each per-class booster encodes the column against
  its own binary target, so every booster sees a category ordering tuned to the
  class it separates.

Feature layout is unchanged (columns are replaced in place), so
`feature_importances_`, `explain`, and `monotone_constraints` keep referring to
the original feature indices.
