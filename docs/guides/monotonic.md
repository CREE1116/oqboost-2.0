# Monotonic constraints

Force the prediction to be non-decreasing (`+1`) or non-increasing (`-1`) in a
feature, holding the others fixed.

```python
# list of length n_features
reg = OQBoostRegressor(monotone_constraints=[1, 0, -1, 0]).fit(X, y)

# or a dict {feature_index: direction}
reg = OQBoostRegressor(monotone_constraints={0: +1, 2: -1}).fit(X, y)
```

`+1` monotone increasing, `-1` decreasing, `0` unconstrained.

## How it works on oblique splits

Holding the other feature fixed, a 2D oblique split `a·u + b·v < t` reduces to a
single threshold on the constrained feature — so the classic axis-tree machinery
(midpoint value-bound propagation down the tree + leaf clamping) ports directly.

When **both** features in a pair are constrained, the split is feasible only if
the sign quadrant agrees: `sign(coefA)·mA == sign(coefB)·mB`. Pairs that violate
this fall back to a 1D split on a feasible feature.

Constraints are enforced exactly — a swept check finds zero violating steps.
