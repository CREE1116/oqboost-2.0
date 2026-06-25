"""Empirical validation of theory.md assumptions.

A. epsilon (low-order ANOVA): XGBoost with interaction order capped at 1 / 2 / full
   (max_depth) — the fraction of signal needing 3rd+ order = (full - 2way)/full.
B. sqrt(log D) generalization: add noise features (D grows), watch OQBoost test
   AUC degrade. Theory predicts O(sqrt(log D / n)) — slow, not sqrt(D).
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "scripts")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
import xgboost as xgb
from oqboost import OQBoostClassifier
from datasets import _load_openml_one, _load_openml_reg

CLF = [(31, "german"), (37, "diabetes"), (1489, "phoneme"), (1067, "kc1"), (44, "spambase")]
REG = [(574, "house_16H"), (308, "puma32H"), (227, "cpu_small")]

print("=" * 72)
print("A. epsilon (is data <=2nd-order ANOVA?)  metric: clf=AUC, reg=R2")
print("   order capped via XGBoost max_depth: 1=main, 2=2-way, 6=full")
print("=" * 72)
print(f"{'dataset':14s} {'d1(main)':>9s} {'d2(2way)':>9s} {'d6(full)':>9s} "
      f"{'eps=(full-2way)/full':>22s}")
def xgb_score(depth, Xtr, ytr, Xte, yte, clf):
    common = dict(n_estimators=400, learning_rate=0.05, max_depth=depth,
                  subsample=0.8, colsample_bytree=0.8, verbosity=0)
    if clf:
        m = xgb.XGBClassifier(**common).fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    m = xgb.XGBRegressor(**common).fit(Xtr, ytr)
    return r2_score(yte, m.predict(Xte))

for did, name in CLF + REG:
    clf = (did, name) in CLF
    if clf:
        _, X, y = _load_openml_one(did, name, 6000)
    else:
        _, X, y = _load_openml_reg(did, name, 6000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=(y if clf else None), random_state=0)
    s1 = xgb_score(1, Xtr, ytr, Xte, yte, clf)
    s2 = xgb_score(2, Xtr, ytr, Xte, yte, clf)
    s6 = xgb_score(6, Xtr, ytr, Xte, yte, clf)
    base = max(s6, 1e-6)
    eps = max(0.0, (s6 - s2)) / base
    print(f"{name:14s} {s1:9.4f} {s2:9.4f} {s6:9.4f} {eps*100:20.1f}%")

print()
print("=" * 72)
print("B. sqrt(log D): add k noise features, OQBoost test AUC vs D")
print("   theory: estimation err ~ sqrt(log D / n) -> slow degrade, not sqrt(D)")
print("=" * 72)
for did, name in [(1489, "phoneme"), (1067, "kc1"), (44, "spambase")]:
    _, X, y = _load_openml_one(did, name, 6000)
    rng = np.random.default_rng(0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    d0 = X.shape[1]
    row = []
    for k in [0, 20, 100, 400]:
        if k:
            Ntr = rng.standard_normal((len(Xtr), k)); Nte = rng.standard_normal((len(Xte), k))
            Xt = np.column_stack([Xtr, Ntr]); Xv = np.column_stack([Xte, Nte])
        else:
            Xt, Xv = Xtr, Xte
        a = roc_auc_score(yte, OQBoostClassifier(n_estimators=150, random_state=0)
                          .fit(Xt, ytr).predict_proba(Xv)[:, 1])
        row.append((d0 + k, a))
    base = row[0][1]
    print(f"{name:10s} (d0={d0}):  " +
          "  ".join(f"D={D}:{a:.4f}({a-base:+.4f})" for D, a in row))

# ── C. LOB on high-eps datasets (recovers the higher-order terms 2D misses) ──
print()
print("=" * 72)
print("C. LOB on high-eps sets: max_lineage 0/2/4 R2 vs CatBoost")
print("   theory: gain should track eps (high-order interaction content)")
print("=" * 72)
try:
    from oqboost import OQBoostRegressor
    from catboost import CatBoostRegressor
    print(f"{'dataset':11s} {'eps':>5s} | {'ml=0':>8s} {'ml=2':>8s} {'ml=4':>8s} | {'CatBoost':>8s}")
    for name, did, eps in [("puma32H", 308, 0.287), ("house_16H", 574, 0.093),
                           ("cpu_small", 227, 0.004)]:
        _, X, y = _load_openml_reg(did, name, 8000)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
        def oq(ml):
            return r2_score(yte, OQBoostRegressor(n_estimators=300, learning_rate=0.05,
                            max_depth=5, max_lineage=ml, n_screen=16, random_state=0)
                            .fit(Xtr, ytr).predict(Xte))
        cat = r2_score(yte, CatBoostRegressor(n_estimators=600, learning_rate=0.05,
                       depth=6, verbose=0, random_state=0).fit(Xtr, ytr).predict(Xte))
        print(f"{name:11s} {eps*100:4.0f}% | {oq(0):8.4f} {oq(2):8.4f} {oq(4):8.4f} | {cat:8.4f}")
except ImportError:
    print("  (skipped: catboost not installed)")
