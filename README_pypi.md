# OQBoost 2.0

**Gradient-boosted 2D-oblique decision trees — histogram-binned, C++ backend.**

OQBoost splits on **oblique hyperplanes over feature pairs** (`a·u + b·v < t`) instead
of axis-aligned thresholds, capturing diagonal and interaction boundaries that
XGBoost/LightGBM approximate with coarse staircases. Version 2.0 is a ground-up
redesign: a histogram-binned 2D-oblique core with a deterministic direction fit.

scikit-learn compatible · compiled C++ (pybind11) + OpenMP.

---

## Install

```bash
pip install oqboost
```

A prebuilt wheel is provided for macOS (arm64, CPython 3.12). On other platforms pip
builds from source — needs a C++17 compiler (`clang++`/`g++`) and, for parallelism,
OpenMP (`brew install libomp` on macOS).

## Quickstart

```python
from oqboost import OQBoostClassifier, OQBoostRegressor

# Binary classification
clf = OQBoostClassifier(
    n_estimators=120, learning_rate=0.06, max_depth=4,
    max_bins=16, subsample=0.8, colsample=0.8, random_state=42,
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]   # P(class 1)
pred  = clf.predict(X_test)

# Regression (squared error)
reg = OQBoostRegressor(n_estimators=120, learning_rate=0.06)
reg.fit(X_train, y_train)
y_hat = reg.predict(X_test)
```

Both are drop-in scikit-learn estimators — usable in `Pipeline`, `GridSearchCV`,
`cross_val_score`, and `clone`.

## Key hyperparameters

| Param | Default | Meaning |
|-------|---------|---------|
| `n_estimators` | 120 | boosting rounds |
| `learning_rate` | 0.06 | shrinkage |
| `max_depth` | 4 | interaction depth (stacked 2D cuts) |
| `max_bins` | 16 | grid / direction-seed resolution (keep small) |
| `subsample` | 0.8 | rows per tree (key overfit lever) |
| `colsample` | 0.8 | features per node |
| `reg_lambda` | 1.0 | L2 regularization |
| `n_screen` | -1 | SIS top-m feature screening (-1 = exhaustive) |

## Why oblique

Axis-aligned boosters need many stacked cuts to approximate a diagonal boundary. On a
2D **XOR** problem XGBoost reaches only AUC ≈ 0.53 while OQBoost reaches ≈ 0.92; on a
**Spiral** OQBoost draws the smoothest boundary of all four boosters. Across tuned
benchmarks it ranks above XGBoost and LightGBM.

Full benchmarks, decision-boundary figures, and design notes:
**https://github.com/cree1116/oqboost**

---

MIT License.
