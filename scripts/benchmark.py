"""
benchmark.py — AnchorTree(cpp) vs XGBoost / LightGBM / CatBoost
실제 OpenML + 합성 데이터에서 held-out test ROC-AUC + 학습시간.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))         # scripts/ (datasets, models)
sys.path.insert(0, str(Path(__file__).parent.parent))  # 프로젝트 루트 (anchor_tree)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from datasets import load_real, SYNTH_2D
from models import make_models

SEED = 42


def eval_all(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    auc, sec = {}, {}
    for name, mdl in make_models(SEED).items():
        t0 = time.perf_counter()
        mdl.fit(Xtr, ytr)
        sec[name] = time.perf_counter() - t0
        auc[name] = roc_auc_score(yte, mdl.predict_proba(Xte)[:, 1])
    return auc, sec


def main():
    names = list(make_models().keys())
    datasets = [(n, X, y) for n, X, y in load_real()]
    for n, fn in SYNTH_2D.items():
        X, y = fn(); datasets.append((n + "(2D)", X, y))

    auc_rows, sec_rows = [], []
    for name, X, y in datasets:
        a, s = eval_all(X, y)
        auc_rows.append({"dataset": name, **a})
        sec_rows.append({"dataset": name, **s})
        print(f"  {name:18s} " + "  ".join(f"{m}={a[m]:.4f}" for m in names))

    auc = pd.DataFrame(auc_rows).set_index("dataset")
    sec = pd.DataFrame(sec_rows).set_index("dataset")

    print("\n" + "=" * 90)
    print("Test ROC-AUC")
    print("=" * 90); print(auc.round(4).to_string())
    print("\nmean AUC rank (1=best):")
    print(auc.rank(axis=1, ascending=False).mean().sort_values().round(2).to_string())
    print("\n" + "=" * 90)
    print("Train time (s)")
    print("=" * 90); print(sec.round(3).to_string())
    print("\nmean train time:")
    print(sec.mean().sort_values().round(3).to_string())

    out = Path(__file__).parent / "output"; out.mkdir(exist_ok=True)
    auc.to_csv(out / "benchmark_auc.csv"); sec.to_csv(out / "benchmark_time.csv")
    print(f"\n  → {out}/benchmark_auc.csv")


if __name__ == "__main__":
    main()