# Early stopping

Set `n_iter_no_change` to stop boosting once a held-out validation metric stops
improving (scikit-learn `GradientBoosting` convention).

```python
clf = OQBoostClassifier(
    n_estimators=2000,        # an upper bound; training usually stops earlier
    n_iter_no_change=20,      # patience: rounds without improvement
    validation_fraction=0.1,  # held out from X (stratified for classification)
    tol=1e-4,                 # min deviance improvement that counts as progress
).fit(X, y)

clf.best_iteration_           # round kept (ensemble is truncated to it)
```

`fit` holds out `validation_fraction` of the training data, and after each round
the backend evaluates the validation **deviance** (logloss for classification,
MSE for regression) on an incrementally-maintained score vector. When it fails to
improve by more than `tol` for `n_iter_no_change` rounds, training stops and the
ensemble is truncated to the best round (`best_iteration_`).

- Multiclass (OvR): each one-vs-rest booster stops independently;
  `best_iteration_` is a list, one per class.
- Disabled under `warm_start` (continuation has no fresh validation split).
- `n_iter_no_change=None` (default) disables early stopping — all
  `n_estimators` rounds run.

## sample_weight

`fit(X, y, sample_weight=w)` scales each sample's gradient and hessian by its
weight, so its influence on the loss is proportional to `w` (exact Newton
weighting). Note this is **not exactly equivalent to repeating rows** — histogram
bin edges are computed unweighted, a second-order difference that shrinks as
`max_bins` grows. `class_weight` (classifier) is folded into `sample_weight`; see
the [classifier API](../api/classifier.md).
