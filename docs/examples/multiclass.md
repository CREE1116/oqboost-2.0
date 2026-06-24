# Example: multiclass classification

`OQBoostClassifier` detects >2 classes and uses one-vs-rest automatically.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from oqboost import OQBoostClassifier

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

clf = OQBoostClassifier(n_estimators=120, max_depth=3).fit(Xtr, ytr)

P = clf.predict_proba(Xte)          # (n, 3), rows sum to 1
pred = clf.predict(Xte)
print("classes:", clf.classes_)
print("ACC :", accuracy_score(yte, pred))
print("mAUC:", roc_auc_score(yte, P, multi_class="ovr", average="macro"))
```

## Class weighting

```python
clf = OQBoostClassifier(class_weight="balanced").fit(Xtr, ytr)
```

## Per-class explanation

```python
phi = clf.explain(Xte[:10])         # (10, 3, n_features)
phi_class0 = phi[:, 0, :]           # contributions to class 0's OvR score

import oqboost.plot as oqp
oqp.plot_importance(clf, class_idx=1)   # importance for class 1
```

See [multiclass guide](../guides/multiclass.md), [explainability](../explainability.md).
