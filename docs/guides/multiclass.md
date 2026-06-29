# Multiclass classification

`OQBoostClassifier` handles more than two classes automatically. Two strategies
are available via the `multiclass` parameter:

- **`"joint"`** (default) — a single softmax model: one shared 2D-oblique tree per
  boosting round with per-class leaf weights, trained on the full K-dimensional
  softmax gradient. Calibrated `predict_proba` straight from the softmax, faster
  (one booster), and the strongest option on most datasets.
- **`"ovr"`** — one-vs-rest: one binary oblique booster per class, with
  probabilities normalized across classes. Useful when per-class patterns differ
  strongly; trains K boosters.

```python
clf = OQBoostClassifier(n_estimators=120).fit(X, y)   # y has >2 classes → joint
clf.classes_                       # the class labels
P = clf.predict_proba(X)           # (n_samples, n_classes), rows sum to 1
pred = clf.predict(X)              # argmax over classes

ovr = OQBoostClassifier(multiclass="ovr").fit(X, y)   # opt into one-vs-rest
```

Binary vs multiclass is detected from `y`; no other configuration is required.
For accuracy keep the default `fast_dir="full"` (all-pair 2D search); switch to
`fast_dir="fast"` (Star anchor) only when the feature count makes the O(d²)
search too slow.

## joint vs ovr

- **Probabilities** — joint emits a native softmax; ovr row-normalizes the
  per-class binary scores.
- **Class baseline** — joint inits each class to its (sample-weighted) log-prior,
  so imbalanced targets and `class_weight` start from the right offset instead of
  a uniform 1/K.
- **`decision_threshold_`** tuning (`threshold="balanced"`/`"f1"`) applies to the
  binary case only; multiclass uses argmax.
- **Early stopping** — joint stops the single model on the multiclass deviance
  (`best_iteration_` is an int); ovr stops each booster independently
  (`best_iteration_` is a list).

## Explainability

`explain(X)` is supported in **`ovr`** mode and returns
`(n_samples, n_classes, n_features)` — `[:, k, :]` are the additive contributions
to class `k`'s score. The [plotting](../api/plotting.md) functions take
`class_idx=k` to view one class. See [explainability](../explainability.md).
`joint` mode shares one tree across classes, so per-class path attribution is not
defined — use `multiclass="ovr"` if you need `explain()`.