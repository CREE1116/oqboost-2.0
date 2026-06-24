# `OQBoostRegressor`

2D-oblique gradient-boosted decision trees for regression (squared error by
default, robust losses available). scikit-learn compatible.

```python
from oqboost import OQBoostRegressor
```

## Parameters

Shares the core tree/boosting parameters with
[`OQBoostClassifier`](classifier.md) (`n_estimators`, `learning_rate`,
`max_depth`, `max_bins`, `reg_lambda`, `min_samples`, `n_screen`, `subsample`,
`colsample`, `fast_dir`, `monotone_constraints`, `categorical_features`,
`max_lineage`, `warm_start`, `n_iter_no_change`, `validation_fraction`, `tol`,
`random_state`). Regression-specific:

| Param | Default | Meaning |
|-------|---------|---------|
| `loss` | `"squared"` | `"squared"` (L2), `"huber"` (robust), `"quantile"` (pinball) |
| `alpha` | 0.9 | huber: delta quantile · quantile: target quantile |
| `clip` | `False` | clamp predictions to the training target range |

`huber` and `quantile` initialize from the median. `quantile` recomputes each
leaf as the `alpha`-quantile of its residuals (line search), so leaf values are
correct pinball minimizers, not `-G/H`.

`threshold` / `class_weight` are classification-only and ignored here.

## Methods

- `fit(X, y, sample_weight=None)`
- `predict(X)` → real-valued predictions.
- `explain(X)` → `(n_samples, n_features)` additive contributions; see [explainability](../explainability.md).

## Attributes

- `n_features_in_`, `feature_names_in_`
- `best_iteration_` — set when early stopping is on
- `feature_importances_`, `coefficient_importances_`, `interaction_importances_`
