# `OQBoostClassifier`

2D-oblique gradient-boosted decision trees for classification. Binary is native
(logistic); multiclass is one-vs-rest. scikit-learn compatible.

```python
from oqboost import OQBoostClassifier
```

## Parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `n_estimators` | 120 | boosting rounds |
| `learning_rate` | 0.06 | shrinkage |
| `max_depth` | 4 | interaction depth (stacked 2D cuts) |
| `max_bins` | 16 | histogram resolution (keep small) |
| `reg_lambda` | 1.0 | L2 regularization |
| `min_samples` | 10 | min samples to split a node |
| `n_screen` | -1 | SIS top-m feature screening (-1 = exhaustive) |
| `subsample` | 0.8 | rows sampled per tree |
| `colsample` | 0.8 | features sampled per node |
| `fast_dir` | 1 | direction finder: 1 = H-weighted gradient regression, 0 = BHC seed (legacy) |
| `threshold` | `"0.5"` | binary decision cut — `"balanced"` / `"f1"` tune it on a holdout |
| `class_weight` | `None` | `"balanced"` or `{class: weight}` dict |
| `monotone_constraints` | `None` | per-feature monotonicity, list of `-1/0/+1` or `{idx: dir}` — see [guide](../guides/monotonic.md) |
| `categorical_features` | `None` | indices / bool mask → cross-fitted target encoding — see [guide](../guides/categorical.md) |
| `max_lineage` | 0 | LOB (experimental) — see [internals/lob](../internals/lob.md) |
| `warm_start` | `False` | add trees incrementally — see [guide](../guides/warm_start.md) |
| `n_iter_no_change` | `None` | early stopping patience — see [guide](../guides/early_stopping.md) |
| `validation_fraction` | 0.1 | held-out fraction for early stopping |
| `tol` | 1e-4 | min validation improvement for early stopping |
| `random_state` | 42 | seed |

## Methods

- `fit(X, y, sample_weight=None)` — `y` binary or multiclass. See [sample_weight notes](../guides/early_stopping.md) on weighting.
- `predict(X)` → class labels.
- `predict_proba(X)` → `(n_samples, n_classes)`, rows sum to 1.
- `explain(X)` → additive contributions; `(n, d)` binary, `(n, n_classes, d)` multiclass. See [explainability](../explainability.md).

## Attributes

- `classes_`, `n_features_in_`, `feature_names_in_`
- `decision_threshold_` — the binary cut actually used
- `best_iteration_` — set when early stopping is on
- `feature_importances_`, `coefficient_importances_`, `interaction_importances_`
