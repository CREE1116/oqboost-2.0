# Multiclass classification

`OQBoostClassifier` handles more than two classes automatically via
**one-vs-rest** (OvR): one binary oblique booster per class, with probabilities
normalized across classes.

```python
clf = OQBoostClassifier(n_estimators=120).fit(X, y)   # y has >2 classes
clf.classes_                       # the class labels
P = clf.predict_proba(X)           # (n_samples, n_classes), rows sum to 1
pred = clf.predict(X)              # argmax over classes
```

No special configuration is required — binary vs multiclass is detected from `y`.

## What's OvR vs native

- Probabilities are row-normalized across the per-class scores.
- `decision_threshold_` tuning (`threshold="balanced"`/`"f1"`) applies to the
  binary case only; multiclass uses argmax.
- Early stopping runs per class (each booster stops independently);
  `best_iteration_` is a list.

A native softmax (single K-output model) is a possible future addition; OvR is
the current implementation and reaches accuracy 1.0 on iris.

## Explainability

`explain(X)` returns `(n_samples, n_classes, n_features)` for multiclass —
`[:, k, :]` are the additive contributions to class `k`'s OvR score. The
[plotting](../api/plotting.md) functions take `class_idx=k` to view one class.
See [explainability](../explainability.md).
