# OQBoost 2.0

**Gradient-boosted 2D-oblique decision trees — histogram-binned, C++ backend.**

OQBoost replaces axis-aligned splits with **oblique hyperplanes over feature pairs**
(`a·u + b·v < t`), capturing diagonal and interaction boundaries that axis-aligned
boosters approximate with coarse staircases. Version 2.0 is a ground-up redesign: a
histogram-binned 2D-oblique core that finds split directions via a BHC-seeded
H-weighted least-squares fit — no random projections, no numerical search.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

<p align="center">
  <img src="docs/decision_boundary.png" alt="OQBoost decision boundaries vs XGBoost / LightGBM / CatBoost" width="820">
</p>

Decision boundaries on synthetic 2D problems. OQBoost draws **smooth diagonal**
boundaries (Spiral, XOR) where axis-aligned XGBoost collapses into blocky staircases.

---

## Key properties

| Feature | OQBoost 2.0 |
|---------|-------------|
| Split type | Oblique — linear combination of **two** features per node |
| Direction finding | BHC-seeded H-weighted least squares (2×2, O(1)) — deterministic |
| Higher-order interactions | Composed via tree depth + boosting (2D atoms) |
| Categorical features | Integer codes through the oblique path (no special encoding) |
| Speed | Global histogram binning + OpenMP-parallel pair search |
| Tasks | `OQBoostClassifier` (binary) · `OQBoostRegressor` |
| API | scikit-learn compatible |
| Backend | Compiled C++ (pybind11) |

---

## Install

```bash
pip install oqboost
```

Building from source needs `clang++`/`g++` (C++17) and, for parallelism, OpenMP
(`brew install libomp` on macOS).

## Quickstart

```python
from oqboost import OQBoostClassifier, OQBoostRegressor

clf = OQBoostClassifier(n_estimators=120, learning_rate=0.06,
                        max_depth=4, subsample=0.8, colsample=0.8)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

reg = OQBoostRegressor().fit(X_train, y_train)
yhat = reg.predict(X_test)
```

Both are drop-in scikit-learn estimators (`get_params`/`set_params`/`clone`,
Pipelines, GridSearchCV).

---

## Benchmark

Optuna-tuned (each model gets the same trial budget), diverse OpenML binary datasets,
held-out test ROC-AUC. Reproduce with:

```bash
python scripts/benchmark_optuna.py 30 15      # tunes all 4 models, caches best params
```

Best params are cached to `docs/optuna_params.json` and reused on re-runs (pass
`--retune` to re-search), so the table below is fully reproducible.

<p align="center">
  <img src="docs/benchmark_optuna.png" alt="Optuna-tuned test AUC across OpenML datasets" width="820">
</p>

OQBoost is strongest on oblique/interaction structure — e.g. 2D **XOR** where
axis-aligned XGBoost collapses to AUC ≈ 0.53 while OQBoost reaches ≈ 0.92, and
**Spiral** where it draws the smoothest boundary of all four boosters (figure above).

---

## How it works

1. **Newton boosting** (logistic / squared-error). Per round, fit one oblique tree to
   the gradient/hessian.
2. **Histogram binning** once at fit: per-feature quantile bins precomputed, so node
   split search is sort-free O(n) accumulation.
3. **2D-oblique split**: for each feature pair, accumulate G/H on the bin grid, seed a
   binary partition by leaf-weight, fit a direction with H-weighted least squares, then
   scan the projection for the threshold. Best of 1D vs 2D by gain.
4. Higher-order interactions come from **depth + boosting**, not wider splits — 2D is
   the bias/variance and search-cost sweet spot.

See [`docs/MODEL.md`](docs/MODEL.md) and [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Key hyperparameters

| Param | Default | Meaning |
|-------|---------|---------|
| `n_estimators` | 120 | boosting rounds |
| `learning_rate` | 0.06 | shrinkage |
| `max_depth` | 4 | interaction depth |
| `max_bins` | 16 | grid / direction-seed resolution (keep small) |
| `subsample` | 0.8 | rows per tree |
| `colsample` | 0.8 | features per node |
| `reg_lambda` | 1.0 | L2 |
| `n_screen` | -1 | SIS top-m feature screening (-1 = exhaustive) |

---

## License

MIT.