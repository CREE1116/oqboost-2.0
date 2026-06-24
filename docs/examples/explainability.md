# Example: explainability

```python
import numpy as np
from oqboost import OQBoostRegressor
import oqboost.plot as oqp

# known structure: x0*x1 interaction + x2 (+) and x3 (-) main effects
rng = np.random.RandomState(1)
X = rng.standard_normal((3000, 7))
y = 1.6 * X[:, 0] * X[:, 1] + 1.1 * X[:, 2] - 0.9 * X[:, 3] + rng.randn(3000) * 0.3
names = ["age", "income", "capital", "debt", "hours", "noise1", "noise2"]

reg = OQBoostRegressor(n_estimators=200, max_depth=4).fit(X, y)
```

## Global importance and interactions

```python
reg.feature_importances_        # gain per feature
reg.coefficient_importances_    # gain * |coef| per feature
reg.interaction_importances_    # d x d learned feature-pair matrix

oqp.plot_importance(reg)        # expect capital / debt to rank high
oqp.plot_interactions(reg)      # expect the (age, income) cell to light up
```

## Per-sample additive attribution

```python
phi = reg.explain(X[:1])        # (1, 7)
# additive: contributions sum to prediction - base
assert np.allclose(phi.sum(1), reg.predict(X[:1]) - y.mean(), atol=1e-6)

oqp.plot_explanation(reg, X[0])          # one-sample waterfall
oqp.plot_explanation_summary(reg, X[:500])  # beeswarm over samples
```

## Comparing to SHAP

Because `explain` is additive (`phi.sum(-1) == pred - base`), the values line up
directly with `shap` values computed for other models. Global rankings agree with
KernelSHAP/TreeSHAP at Spearman ~0.71–0.77.

See [explainability](../explainability.md), [plotting API](../api/plotting.md).
