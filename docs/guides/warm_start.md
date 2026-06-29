# Warm start (incremental training)

With `warm_start=True`, raising `n_estimators` and re-fitting on the same data
adds only the new trees instead of retraining from scratch.

```python
clf = OQBoostClassifier(n_estimators=100, warm_start=True).fit(X, y)
clf.set_params(n_estimators=200).fit(X, y)   # trains only the additional 100 trees
clf.set_params(n_estimators=300).fit(X, y)   # +100 more
```

The boosting state (the running raw scores) is reconstructed from the stored
trees, so nothing extra needs to be persisted between calls; the RNG is carried
on the booster so the added rounds continue the same sequence. With
`subsample=1.0`, incrementally grown trees are bit-identical to training the full
ensemble from scratch.

Works for binary, regression, and multiclass — both `joint` (the single softmax
model grows) and `ovr` (each class booster grows).

## Notes

- Lowering `n_estimators` under `warm_start` does **not** drop trees (it is a
  no-op); start a fresh estimator to shrink.
- Early stopping is disabled while warm-starting.
- The input must have the same number of features as the initial fit.
