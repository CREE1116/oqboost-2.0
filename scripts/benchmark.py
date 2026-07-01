"""
benchmark.py — test 평가 (튜닝과 분리). binary / multiclass / regression.

`optimize.py`가 만든 `docs/optuna_params.json`의 best_params를 읽어 각 모델을
train에 적합하고 held-out test에서 비교한다. 튜닝은 하지 않는다.

메트릭:
  - binary:     ROC-AUC, accuracy, balanced accuracy (val서 threshold 튜닝, 전 모델 동일)
  - multiclass: accuracy (argmax, primary) + OvR macro-AUC, balanced accuracy
  - regression: R², RMSE, MAE
  - 공통: train time, inference time

사용: python optimize.py ... ; python benchmark.py [--tasks binary,multiclass,regression]
출력: docs/benchmark.csv (long), docs/images/benchmark_optuna.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tuning
import json, time
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             r2_score, mean_squared_error, mean_absolute_error)

from datasets import SUITES, CAT_INDEX
from tuning import models_for, COLORS, PARAMS_JSON, build, split_tvt, model_inputs, REGISTRY

_tflag = [a for a in sys.argv if a.startswith("--tasks")]
TASKS = (_tflag[0].split("=")[1].split(",") if _tflag and "=" in _tflag[0]
         else ["binary", "multiclass", "regression"])
NO_CAT = "--no-categorical" in sys.argv   # 기본: 네이티브 범주 ON
# multiclass primary = accuracy (argmax) to match the tuning objective; binary
# stays AUC, regression R². (auc/acc/bacc are all still reported per task.)
PRIMARY = {"binary": "auc", "multiclass": "acc", "regression": "r2"}


def _infer_time(m, X, reps=5, proba=True):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        m.predict_proba(X) if proba else m.predict(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _best_threshold(y, proba):
    """val서 balanced accuracy 극대화 cut (불균형서 0.5는 다수class로 붕괴)."""
    cands = np.unique(np.quantile(proba, np.linspace(0.02, 0.98, 49)))
    sc = [balanced_accuracy_score(y, (proba >= t).astype(int)) for t in cands]
    return float(cands[int(np.argmax(sc))])


def eval_classification(task, m, Xtr, ytr, Xva, yva, Xtt, ytt):
    t0 = time.perf_counter(); m.fit(Xtr, ytr); train_s = time.perf_counter() - t0
    if task == "binary":
        thr = _best_threshold(yva, m.predict_proba(Xva)[:, 1])
        proba = m.predict_proba(Xtt)[:, 1]
        pred = (proba >= thr).astype(int)
        auc = roc_auc_score(ytt, proba)
    else:  # multiclass
        P = m.predict_proba(Xtt)
        pred = m.classes_[P.argmax(1)] if hasattr(m, "classes_") else P.argmax(1)
        auc = roc_auc_score(ytt, P, multi_class="ovr", average="macro")
    return dict(auc=auc, acc=accuracy_score(ytt, pred),
                bacc=balanced_accuracy_score(ytt, pred),
                train_s=train_s, infer_s=_infer_time(m, Xtt, proba=True))


def eval_regression(m, Xtr, ytr, Xtt, ytt):
    t0 = time.perf_counter(); m.fit(Xtr, ytr); train_s = time.perf_counter() - t0
    yhat = m.predict(Xtt)
    return dict(r2=r2_score(ytt, yhat),
                rmse=mean_squared_error(ytt, yhat) ** 0.5,
                mae=mean_absolute_error(ytt, yhat),
                train_s=train_s, infer_s=_infer_time(m, Xtt, proba=False))


def main():
    if not PARAMS_JSON.exists():
        sys.exit(f"[!] {PARAMS_JSON} 없음. 먼저 `python optimize.py` 실행.")
    cache = json.loads(PARAMS_JSON.read_text())

    rows = []
    for task in TASKS:
        strat = task != "regression"
        for name, X, y in SUITES[task](max_rows=10000):
            print(f"Running [{task}] {name}...", flush=True)
            Xtr, Xva, Xtt, ytr, yva, ytt = split_tvt(X, y, stratify=strat)
            cat = [] if NO_CAT else CAT_INDEX.get(name, [])
            cards = {j: int(round(X[:, j].max())) + 1 for j in cat}
            for mname in models_for(task):
                params = cache.get(name, {}).get(mname) if name in cache else None
                print(f"  Fitting {mname}...", flush=True)
                if params is None:
                    _, Model = REGISTRY[task][mname]
                    kw = {}
                    from tuning import inject_categorical
                    inject_categorical(mname, kw, cat)
                    m = Model(**kw)
                else:
                    m = build(mname, params, task=task, cat_idx=cat)
                Xtr_m = model_inputs(mname, Xtr, cat, cards)
                Xva_m = model_inputs(mname, Xva, cat, cards)
                Xtt_m = model_inputs(mname, Xtt, cat, cards)
                if task == "regression":
                    met = eval_regression(m, Xtr_m, ytr, Xtt_m, ytt)
                else:
                    met = eval_classification(task, m, Xtr_m, ytr, Xva_m, yva, Xtt_m, ytt)
                rows.append(dict(task=task, dataset=name, model=mname, **met))
            print(f"  done [{task}] {name}  (cat={len(cat)})", flush=True)

    df = pd.DataFrame(rows)
    out = PARAMS_JSON.parent
    df.to_csv(out / "benchmark.csv", index=False)
    _report(df)
    (out / "images").mkdir(parents=True, exist_ok=True); _plot(df, out / "images" / "benchmark_optuna.png")
    print(f"\n  → {out}/benchmark.csv, benchmark_optuna.png")


def _report(df):
    for task in [t for t in TASKS if t in df.task.values]:
        sub = df[df.task == task]
        mets = (["auc", "acc", "bacc"] if task != "regression"
                else ["r2", "rmse", "mae"]) + ["train_s", "infer_s"]
        print("\n" + "#" * 78); print(f"# {task}"); print("#" * 78)
        for met in mets:
            piv = sub.pivot(index="dataset", columns="model", values=met)
            print(f"\n[{met}]"); print(piv.round(4).to_string())
        # primary metric 평균 랭크
        prim = PRIMARY[task]
        piv = sub.pivot(index="dataset", columns="model", values=prim)
        asc = (prim in ("rmse", "mae"))  # 작을수록 좋은 메트릭이면 ascending 랭크
        rank = piv.rank(axis=1, ascending=asc).mean().sort_values()
        print(f"\nmean {prim} rank (1=best):"); print(rank.round(2).to_string())


def _plot(df, path):
    import matplotlib.pyplot as plt
    tasks = [t for t in TASKS if t in df.task.values]
    fig, axes = plt.subplots(len(tasks), 1, figsize=(13, 4.2 * len(tasks)),
                             squeeze=False)
    for r, task in enumerate(tasks):
        ax = axes[r, 0]
        prim = PRIMARY[task]
        sub = df[df.task == task]
        piv = sub.pivot(index="dataset", columns="model", values=prim)
        models = [m for m in ["OQBoost", "XGBoost", "LightGBM", "CatBoost", "ObliqueTree", "ObliqueForest"] if m in piv.columns]
        piv = piv[models]
        ds = list(piv.index); x = np.arange(len(ds)); w = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            ax.bar(x + (i - (len(models) - 1) / 2) * w, piv[m].values, w,
                   label=m, color=COLORS[m], alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(ds, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(prim.upper() if prim != "r2" else "R²")
        rk = piv.rank(axis=1, ascending=(prim in ("rmse", "mae"))).mean()
        best = rk.idxmin()
        ax.set_title(f"{task} — test {prim.upper() if prim!='r2' else 'R²'} "
                     f"(mean-rank best: {best})")
        ax.legend(fontsize=8, ncol=len(models)); ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        if prim == "r2":
            ax.set_ylim(max(-0.1, np.nanmin(piv.values) - 0.05), 1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
