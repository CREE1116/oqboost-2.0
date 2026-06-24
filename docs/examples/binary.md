# Example: binary classification

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from oqboost import OQBoostClassifier

X, y = make_classification(n_samples=4000, n_features=20, n_informative=10,
                           weights=[0.85, 0.15], random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

clf = OQBoostClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                        threshold="balanced").fit(Xtr, ytr)

proba = clf.predict_proba(Xte)[:, 1]
pred = clf.predict(Xte)
print("AUC :", roc_auc_score(yte, proba))
print("bACC:", balanced_accuracy_score(yte, pred))
print("cut :", clf.decision_threshold_)
```

## Imbalanced data

Probabilities are calibrated (mean ≈ base rate), so on imbalanced data the default
0.5 cut collapses balanced accuracy. Either tune the cut (`threshold="balanced"`
above) or reweight:

```python
clf = OQBoostClassifier(class_weight="balanced").fit(Xtr, ytr)
# or per-sample
clf.fit(Xtr, ytr, sample_weight=w)
```

## With early stopping

```python
clf = OQBoostClassifier(n_estimators=2000, n_iter_no_change=20,
                        validation_fraction=0.1).fit(Xtr, ytr)
print("stopped at", clf.best_iteration_)
```

See [classifier API](../api/classifier.md), [early stopping](../guides/early_stopping.md).
