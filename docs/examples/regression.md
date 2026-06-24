# Example: regression

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from oqboost import OQBoostRegressor

X, y = fetch_california_housing(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

reg = OQBoostRegressor(n_estimators=400, learning_rate=0.05, max_depth=5).fit(Xtr, ytr)
yhat = reg.predict(Xte)
print("R2 :", r2_score(yte, yhat))
print("MAE:", mean_absolute_error(yte, yhat))
```

## Robust losses (outliers)

```python
# Huber: quadratic near zero, linear in the tails (delta at the alpha quantile)
reg = OQBoostRegressor(loss="huber", alpha=0.9).fit(Xtr, ytr)

# Clamp predictions to the training target range
reg = OQBoostRegressor(loss="huber", clip=True).fit(Xtr, ytr)
```

## Quantile regression (prediction intervals)

```python
lo = OQBoostRegressor(loss="quantile", alpha=0.1).fit(Xtr, ytr)
hi = OQBoostRegressor(loss="quantile", alpha=0.9).fit(Xtr, ytr)
lower, upper = lo.predict(Xte), hi.predict(Xte)   # ~80% interval
```

Note: extreme quantiles (α near 0/1) bias inward on shallow trees — increase
`max_depth` for tighter tails. The median (α=0.5) is accurate.

See [regressor API](../api/regressor.md).
