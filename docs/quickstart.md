# Quickstart

OQBoost ships two estimators with the standard scikit-learn API:
`OQBoostClassifier` (binary + multiclass) and `OQBoostRegressor`.

```python
from oqboost import OQBoostClassifier, OQBoostRegressor

# Binary / multiclass classification
clf = OQBoostClassifier(n_estimators=120, learning_rate=0.06, max_depth=4)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)   # (n_samples, n_classes), rows sum to 1
pred  = clf.predict(X_test)

# Regression
reg = OQBoostRegressor().fit(X_train, y_train)
yhat = reg.predict(X_test)
```

Both are drop-in scikit-learn estimators — usable in `Pipeline`, `GridSearchCV`,
`cross_val_score`, `clone`, and picklable / joblib-dumpable.

## Inputs

- **NumPy or pandas** — pandas column names are tracked in `feature_names_in_`.
- **scipy sparse** (CSR/CSC) — accepted, densified internally.
- **Missing values** — `NaN` is handled natively (routed to a learned bin); no imputation needed.

## Key hyperparameters

| Param | Default | Meaning |
|-------|---------|---------|
| `n_estimators` | 120 | boosting rounds |
| `learning_rate` | 0.06 | shrinkage |
| `max_depth` | 4 | interaction depth (stacked 2D cuts) |
| `max_bins` | 16 | histogram resolution (keep small) |
| `subsample` | 0.8 | rows per tree |
| `colsample` | 0.8 | features per node |
| `reg_lambda` | 1.0 | L2 regularization |
| `n_screen` | -1 | SIS top-m feature screening (-1 = exhaustive) |

The defaults are deliberately conservative (low `learning_rate`, row/feature
subsampling) and are competitive untuned — see [benchmarks](benchmarks.md).

## Next

- Per-task examples: [binary](examples/binary.md) · [regression](examples/regression.md) · [multiclass](examples/multiclass.md)
- Feature guides: [categorical](guides/categorical.md) · [monotonic](guides/monotonic.md) · [early stopping](guides/early_stopping.md) · [warm start](guides/warm_start.md) · [multiclass](guides/multiclass.md)
- [Explainability](explainability.md) · [API reference](api/classifier.md)
