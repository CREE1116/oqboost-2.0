"""
model_similarity.py — 4개 모델(OQBoost/XGBoost/LightGBM/CatBoost) 예측 유사도.

"OQBoost가 기성 부스터와 얼마나 다른 함수를 학습하나"를 본다. 낮을수록 앙상블
다양성↑ / 사선구조 고유성↑.
  - 분류: 예측 일치도(agreement, 같은 hard label 비율) — task에 맞는 직관적 유사도.
  - 회귀: 예측의 Pearson correlation.
데이터셋들 평균 → task별 유사도 히트맵 → docs/model_similarity.png

캐시된 튜닝 파라미터가 있으면 사용(`--use-cache`), 없으면 합리적 기본값으로 적합.
기본은 sklearn 내장 데이터(네트워크 불필요). `--full`이면 OpenML 스위트 전체.

사용: python scripts/model_similarity.py [--full] [--use-cache]
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import SUITES
from tuning import models_for, build, split_tvt, COLORS, PARAMS_JSON

ORDER = ["OQBoost", "XGBoost", "LightGBM", "CatBoost"]
FULL = "--full" in sys.argv
USE_CACHE = "--use-cache" in sys.argv
OUT = Path(__file__).parent.parent / "docs" / "model_similarity.png"


def _defaults(task):
    """튜닝 캐시 없을 때 쓰는 합리적 기본 (동일 예산 비교)."""
    from oqboost import OQBoostClassifier, OQBoostRegressor
    import xgboost as xgb, lightgbm as lgb
    from catboost import CatBoostClassifier, CatBoostRegressor
    ne = 200
    if task == "regression":
        return {"OQBoost": OQBoostRegressor(n_estimators=ne, n_screen=16, random_state=42),
                "XGBoost": xgb.XGBRegressor(n_estimators=ne, max_depth=5, verbosity=0),
                "LightGBM": lgb.LGBMRegressor(n_estimators=ne, verbose=-1),
                "CatBoost": CatBoostRegressor(n_estimators=ne, verbose=False, allow_writing_files=False)}
    return {"OQBoost": OQBoostClassifier(n_estimators=ne, n_screen=16, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=ne, max_depth=5, verbosity=0),
            "LightGBM": lgb.LGBMClassifier(n_estimators=ne, verbose=-1),
            "CatBoost": CatBoostClassifier(n_estimators=ne, verbose=False, allow_writing_files=False)}


def _models(task, name, cache):
    if USE_CACHE and name in cache:
        out = {}
        for m in ORDER:
            p = cache[name].get(m)
            if p is not None:
                out[m] = build(m, p, task=task)
        if len(out) == len(ORDER):
            return out
    return _defaults(task)


def _preds(task, name, X, y, cache):
    """test 예측 벡터(모델별). 분류=hard label, 회귀=실수."""
    strat = task != "regression"
    Xtr, _, Xtt, ytr, _, ytt = split_tvt(X, y, stratify=strat)
    P = {}
    for m, est in _models(task, name, cache).items():
        est.fit(Xtr, ytr)
        P[m] = est.predict(Xtt)
    return P


def _pairwise(task, preds_list):
    """데이터셋들의 예측을 모아 4×4 유사도 행렬 (평균)."""
    mats = []
    for P in preds_list:
        M = np.eye(4)
        for i, a in enumerate(ORDER):
            for j, b in enumerate(ORDER):
                if j <= i:
                    continue
                
                # CatBoost의 (N, 1) 셰이프 분해 및 안전한 비교를 위해 1차원 평탄화(.ravel()) 처리
                pred_a = np.asarray(P[a]).ravel()
                pred_b = np.asarray(P[b]).ravel()
                
                if task == "regression":
                    s = np.corrcoef(pred_a, pred_b)[0, 1]
                else:
                    s = np.mean(pred_a == pred_b)
                M[i, j] = M[j, i] = s
        mats.append(M)
    return np.mean(mats, axis=0)


def main():
    cache = json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else {}
    tasks = ["binary", "multiclass", "regression"]
    builtin_only = not FULL
    mats = {}
    for task in tasks:
        data = SUITES[task](max_rows=4000,
                            **({"include_builtin": True} if task != "binary"
                               else {"include_breast": True}))
        if builtin_only:
            data = data[:2]   # sklearn 내장 위주 (네트워크 최소)
        preds = []
        for name, X, y in data:
            try:
                preds.append(_preds(task, name, X, y, cache))
                print(f"  [{task}] {name} done")
            except Exception as e:
                print(f"  [skip] {task}/{name}: {e}")
        if preds:
            mats[task] = (_pairwise(task, preds), len(preds))

    # --- 콘솔 수치 출력 ---
    print("\n" + "="*25 + " MODEL SIMILARITY NUMERICAL RESULTS " + "="*25)
    for task, (M, ndata) in mats.items():
        label = "Label Agreement" if task != "regression" else "Pearson Correlation"
        print(f"\n▶ Task: {task.upper()} ({label}, {ndata} datasets)")
        print(f"{'':<12}" + "".join(f"{m:>12}" for m in ORDER))
        for i, m1 in enumerate(ORDER):
            row_str = "".join(f"{M[i, j]:12.4f}" for j in range(4))
            print(f"{m1:<12}{row_str}")
    print("="*86 + "\n")

    _plot(mats)


def _plot(mats):
    tasks = list(mats)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 4.2), squeeze=False)
    for c, task in enumerate(tasks):
        M, ndata = mats[task]
        ax = axes[0, c]
        im = ax.imshow(M, cmap="YlGnBu", vmin=M.min(), vmax=1.0)
        ax.set_xticks(range(4), ORDER, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(4), ORDER, fontsize=8)
        label = "label agreement" if task != "regression" else "pred correlation"
        ax.set_title(f"{task}  ({label}, {ndata} sets)", fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if M[i, j] > 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Pairwise prediction similarity — OQBoost vs XGBoost / LightGBM / CatBoost",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()