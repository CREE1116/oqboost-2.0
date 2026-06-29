# `oqboost.plot`

matplotlib visualizations for OQBoost's native explanations. No `shap` or other
plotting dependency — only matplotlib (install with `pip install oqboost[plot]`).
Lazily imported, so the core package has no hard matplotlib dependency.

```python
import oqboost.plot as oqp
```

All functions take a fitted OQBoost model and return a matplotlib `Axes`
(accepting an `ax=` to compose subplots). For multiclass `ovr` models, pass
`class_idx=k` to select one class (per-class views require `multiclass="ovr"`).

## Functions

### `plot_importance(model, kind="gain", top=None, ax=None, class_idx=None)`
Bar chart of feature importance. `kind="gain"` uses `feature_importances_`;
`kind="coef"` uses `coefficient_importances_` (gain × |coef|).

### `plot_interactions(model, top=None, ax=None, class_idx=None)`
Heatmap of `interaction_importances_` — the `d × d` learned feature-pair matrix.

### `plot_explanation(model, x, top=None, ax=None, class_idx=None)`
Per-sample additive contributions for a single sample `x` (waterfall-style bar);
positive features pushed the prediction up, negative down. Bars sum to
`prediction − base`.

### `plot_explanation_summary(model, X, top=None, ax=None, class_idx=None)`
SHAP-style beeswarm over a sample set `X` — distribution of each feature's
contribution, colored by feature value.

## Example

```python
import matplotlib.pyplot as plt
import oqboost.plot as oqp

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
oqp.plot_importance(model, ax=axes[0, 0])
oqp.plot_interactions(model, ax=axes[0, 1])
oqp.plot_explanation(model, X[0], ax=axes[1, 0])
oqp.plot_explanation_summary(model, X, ax=axes[1, 1])
fig.tight_layout()
```

See [explainability](../explainability.md) for what the underlying quantities mean.
