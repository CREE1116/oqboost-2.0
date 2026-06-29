# OQBoost 2.0

**Gradient-boosted 2D-oblique decision trees — histogram-binned, C++ backend.**

OQBoost splits on **oblique hyperplanes over feature pairs** (`a·u + b·v < t`) instead
of axis-aligned thresholds, so diagonal and interaction boundaries are represented
directly rather than as axis-aligned approximations. Version 2.0 finds split directions
by H-weighted least-squares regression of the gradient — deterministic, one 2×2 solve
per feature pair (no random projections or numerical search).

scikit-learn compatible · compiled C++ (pybind11) + OpenMP · **native missing-value
handling** (NaN routed to a learned bin) · pandas / scipy-sparse inputs.

## Install

```bash
pip install oqboost
```

Prebuilt wheels for Windows, macOS (arm64), and Linux. On other platforms pip builds
from source — needs a C++17 compiler and (for parallelism) OpenMP.

## Quickstart

```python
from oqboost import OQBoostClassifier, OQBoostRegressor

# Binary / multiclass classification (3+ classes use a joint softmax automatically)
clf = OQBoostClassifier(n_estimators=120, learning_rate=0.06, max_depth=4)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)   # (n_samples, n_classes), rows sum to 1
pred  = clf.predict(X_test)

# Regression
reg = OQBoostRegressor().fit(X_train, y_train)
y_hat = reg.predict(X_test)
```

Both are drop-in scikit-learn estimators — usable in `Pipeline`, `GridSearchCV`,
`cross_val_score`, and `clone`; pickle / joblib compatible.

## Documentation

Full documentation is on GitHub: **https://github.com/cree1116/oqboost-2.0**

- [Quickstart](https://github.com/cree1116/oqboost-2.0/blob/main/docs/quickstart.md) · [Benchmarks](https://github.com/cree1116/oqboost-2.0/blob/main/docs/benchmarks.md) · [Explainability](https://github.com/cree1116/oqboost-2.0/blob/main/docs/explainability.md)
- API — [Classifier](https://github.com/cree1116/oqboost-2.0/blob/main/docs/api/classifier.md) · [Regressor](https://github.com/cree1116/oqboost-2.0/blob/main/docs/api/regressor.md) · [Plotting](https://github.com/cree1116/oqboost-2.0/blob/main/docs/api/plotting.md)
- Guides — [Categorical](https://github.com/cree1116/oqboost-2.0/blob/main/docs/guides/categorical.md) · [Monotonic](https://github.com/cree1116/oqboost-2.0/blob/main/docs/guides/monotonic.md) · [Early stopping](https://github.com/cree1116/oqboost-2.0/blob/main/docs/guides/early_stopping.md) · [Warm start](https://github.com/cree1116/oqboost-2.0/blob/main/docs/guides/warm_start.md) · [Multiclass](https://github.com/cree1116/oqboost-2.0/blob/main/docs/guides/multiclass.md)
- Internals — [Algorithm](https://github.com/cree1116/oqboost-2.0/blob/main/docs/internals/algorithm.md) · [LOB](https://github.com/cree1116/oqboost-2.0/blob/main/docs/internals/lob.md) · [Roadmap](https://github.com/cree1116/oqboost-2.0/blob/main/docs/internals/roadmap.md)

> OQBoost 2.0 is a ground-up rewrite. The original 1.x line (oblique splits via a
> Deterministic Gradient-Covariance Scan) lives at
> [cree1116/OQBoost](https://github.com/cree1116/OQBoost).

---

MIT License.
