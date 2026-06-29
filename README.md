# OQBoost 2.0

**Gradient-boosted 2D-oblique decision trees — histogram-binned, C++ backend.**

OQBoost uses **oblique splits over feature pairs** (`a·u + b·v < t`) instead of
axis-aligned thresholds, so diagonal and interaction boundaries are represented
directly rather than as axis-aligned approximations. Version 2.0 is a histogram-binned
2D-oblique core that finds split directions by H-weighted least-squares regression of
the gradient (no random projections or numerical search), with a C++ backend.

> **Lineage:** OQBoost 1.x ([cree1116/OQBoost](https://github.com/cree1116/OQBoost))
> found oblique directions with a Deterministic Gradient-Covariance Scan (DGCS).
> 2.0 is a fresh codebase with a different, faster direction finder and a C++ backend.

[![PyPI version](https://img.shields.io/pypi/v/oqboost.svg)](https://pypi.org/project/oqboost/)
[![Python versions](https://img.shields.io/pypi/pyversions/oqboost.svg)](https://pypi.org/project/oqboost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build](https://github.com/CREE1116/oqboost-2.0/actions/workflows/wheels.yml/badge.svg)](https://github.com/CREE1116/oqboost-2.0/actions/workflows/wheels.yml)
[![Tests](https://github.com/CREE1116/oqboost-2.0/actions/workflows/tests.yml/badge.svg)](https://github.com/CREE1116/oqboost-2.0/actions/workflows/tests.yml)

<p align="center">
  <img src="docs/images/decision_boundary.png" alt="OQBoost decision boundaries vs XGBoost / LightGBM / CatBoost" width="820">
</p>

Decision boundaries on synthetic 2D problems. Because splits are oblique, OQBoost
represents diagonal boundaries (Spiral, XOR) directly rather than approximating
them with axis-aligned steps.

---

## Install

```bash
pip install oqboost
```

Prebuilt wheels for Windows, macOS (arm64), and Linux. See
[docs/installation.md](docs/installation.md) for source builds and OpenMP.

## Quickstart

```python
from oqboost import OQBoostClassifier, OQBoostRegressor

clf = OQBoostClassifier(n_estimators=120, learning_rate=0.06, max_depth=4)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

reg = OQBoostRegressor().fit(X_train, y_train)
yhat = reg.predict(X_test)
```

Drop-in scikit-learn estimators — Pipelines, GridSearchCV, `clone`, pickle/joblib.
More in [docs/quickstart.md](docs/quickstart.md).

## Key properties

| Feature | OQBoost 2.0 |
| ------- | ----------- |
| Split type | Oblique — linear combination of **two** features per node |
| Direction finding | H-weighted gradient regression (2×2, O(1)) — deterministic |
| Higher-order interactions | Composed via tree depth + boosting (2D atoms) |
| Missing values | Native — NaN routed to a learned bin (no imputation) |
| Inputs | NumPy / pandas (feature names) / scipy sparse |
| Tasks | `OQBoostClassifier` (binary + multiclass: joint softmax / OvR) · `OQBoostRegressor` |
| API | scikit-learn compatible (`check_estimator`) |
| Backend | Compiled C++ (pybind11) |

---

## Benchmark

Independently Optuna-tuned across diverse OpenML datasets (binary / multiclass /
regression), held-out test metrics. Mean rank (1 = best), wins in parentheses:

| Task (datasets) | OQBoost | CatBoost | XGBoost | LightGBM |
| --------------- | ------: | -------: | ------: | -------: |
| Binary (12)     | **2.08** (6) | 2.42 (2) | 2.62 (2) | 2.88 (1) |
| Multiclass (10) | **2.20** (3) | **2.20** (3) | 3.20 (0) | 2.40 (1) |
| Regression (10) | 2.50 (2) | **1.40** (6) | 3.20 (1) | 2.90 (1) |

OQBoost leads the binary suite on mean rank and wins, ties CatBoost for the lead
on multiclass, and lands second to CatBoost on regression. It also has the best
mean balanced accuracy on both classification suites (binary 0.856, multiclass
0.826). Full tables, prediction-similarity, and reproduction in
[docs/benchmarks.md](docs/benchmarks.md).

---

## Documentation

- **[Installation](docs/installation.md)** · **[Quickstart](docs/quickstart.md)** · **[Benchmarks](docs/benchmarks.md)** · **[Explainability](docs/explainability.md)**
- **API** — [Classifier](docs/api/classifier.md) · [Regressor](docs/api/regressor.md) · [Plotting](docs/api/plotting.md)
- **Guides** — [Categorical](docs/guides/categorical.md) · [Monotonic](docs/guides/monotonic.md) · [Early stopping](docs/guides/early_stopping.md) · [Warm start](docs/guides/warm_start.md) · [Multiclass](docs/guides/multiclass.md)
- **Internals** — [Algorithm](docs/internals/algorithm.md) · [LOB](docs/internals/lob.md) · [Roadmap](docs/internals/roadmap.md)
- **Examples** — [Binary](docs/examples/binary.md) · [Regression](docs/examples/regression.md) · [Multiclass](docs/examples/multiclass.md) · [Explainability](docs/examples/explainability.md)

---

## License

MIT.
