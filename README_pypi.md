# OQBoost 2.0

**Gradient-boosted 2D-oblique decision trees — histogram-binned, C++ backend.**

OQBoost splits on **oblique hyperplanes over feature pairs** (`a·u + b·v < t`) instead
of axis-aligned thresholds, so diagonal and interaction boundaries are represented
directly rather than as axis-aligned approximations. Version 2.0 is a histogram-binned
2D-oblique core that finds split directions by H-weighted least-squares regression of
the gradient — deterministic, one 2×2 solve per feature pair (no random projections or
numerical search).

scikit-learn compatible · compiled C++ (pybind11) + OpenMP · **native missing-value
handling** (NaN routed to a learned bin, no imputation needed).

---

## Install

```bash
pip install oqboost
```

Prebuilt wheels are provided for Windows, macOS (arm64), and Ubuntu/Linux via
cibuildwheel. On other platforms pip builds from source — needs a C++17 compiler
(`clang++`/`g++`) and, for parallelism, OpenMP (`brew install libomp` on macOS).

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

## More features

```python
# Imbalanced data: tune the decision threshold on a holdout (probabilities stay
# calibrated; "balanced" maximizes balanced accuracy, "f1" maximizes F1).
clf = OQBoostClassifier(threshold="balanced").fit(X_train, y_train)
clf.decision_threshold_          # the chosen cut

# Robust regression: huber / quantile losses (init = median), optional output clip.
reg = OQBoostRegressor(loss="huber", alpha=0.9, clip=True).fit(X_train, y_train)
q90 = OQBoostRegressor(loss="quantile", alpha=0.9).fit(X_train, y_train)  # 90th pctile

# Monotonic constraints (-1 / 0 / +1 per feature; list or {index: dir} dict),
# enforced through the oblique splits.
reg = OQBoostRegressor(monotone_constraints={0: +1, 3: -1}).fit(X_train, y_train)

# Incremental training: add trees without refitting from scratch.
clf = OQBoostClassifier(n_estimators=100, warm_start=True).fit(X_train, y_train)
clf.set_params(n_estimators=200).fit(X_train, y_train)   # trains only +100 trees

# Serialization: models are pickle / joblib compatible out of the box.
import pickle, joblib
pickle.dump(clf, open("clf.pkl", "wb"))
clf2 = pickle.load(open("clf.pkl", "rb"))
joblib.dump(clf, "clf.joblib")

# Native explanations (no SHAP dependency)
clf.feature_importances_         # Σ gain per feature
clf.coefficient_importances_     # Σ gain·|coef| (direction-weighted)
clf.interaction_importances_     # d×d matrix, Σ gain·|a|·|b| — learned feature pairs
phi = clf.explain(X_test)        # (n, n_features) additive contributions
                                 # phi.sum(axis=1) == raw prediction − base (like SHAP)

# Plots (pip install oqboost[plot])
import oqboost.plot as oqp
oqp.plot_importance(clf)              # gain / gain·|coef| bar
oqp.plot_interactions(clf)           # pairwise-interaction heatmap
oqp.plot_explanation(clf, x)         # one-sample additive contributions
oqp.plot_explanation_summary(clf, X) # SHAP-style beeswarm
```

## Key hyperparameters

| Param                  | Default     | Meaning                                                                           |
| ---------------------- | ----------- | --------------------------------------------------------------------------------- |
| `n_estimators`         | 120         | boosting rounds                                                                   |
| `learning_rate`        | 0.06        | shrinkage                                                                         |
| `max_depth`            | 4           | interaction depth (stacked 2D cuts)                                               |
| `max_bins`             | 16          | grid / direction-seed resolution (keep small)                                     |
| `subsample`            | 0.8         | rows per tree (key overfit lever)                                                 |
| `colsample`            | 0.8         | features per node                                                                 |
| `reg_lambda`           | 1.0         | L2 regularization                                                                 |
| `n_screen`             | -1          | SIS top-m feature screening (-1 = exhaustive)                                     |
| `threshold`            | `"0.5"`     | binary decision cut — `"balanced"`/`"f1"` tunes it on a holdout (imbalanced data) |
| `loss`                 | `"squared"` | regression loss — `"huber"`/`"quantile"` are outlier-robust                       |
| `alpha`                | 0.9         | huber δ-quantile / quantile target                                                |
| `clip`                 | `False`     | clamp regression output to training target range                                  |
| `monotone_constraints` | `None`      | per-feature monotonicity `-1`/`0`/`+1` (list or `{idx: dir}` dict)                |
| `warm_start`           | `False`     | add trees on top of the existing model when `n_estimators` grows (incremental)     |
| `categorical_features` | `None`      | indices / bool mask of categorical columns → lossless binning (no level merging)  |

## Why oblique

Axis-aligned boosters need several stacked cuts to approximate a diagonal boundary,
while an oblique split represents it directly. On 2D problems (XOR, Spiral,
Checkerboard) OQBoost draws the diagonal boundary with oblique cuts rather than
axis-aligned steps. On Optuna-tuned tabular benchmarks it is competitive with the
established gradient-boosting libraries.

Full benchmarks, decision-boundary figures, and design notes:
**https://github.com/cree1116/oqboost-2.0**

> **Note:** OQBoost 2.0 is a ground-up rewrite. The original 1.x line — oblique splits
> via a Deterministic Gradient-Covariance Scan (DGCS) — lives at
> [cree1116/OQBoost](https://github.com/cree1116/OQBoost). 2.0 replaces the direction
> finder with a histogram-binned H-weighted gradient-regression fit (one 2×2 solve per pair).

---

MIT License.
