"""
optimize.py — Optuna 하이퍼파라미터 튜닝 (벤치마크와 분리)

binary·multiclass·regression 세 task의 다양한 OpenML 데이터셋에서 4개 모델
(OQBoost/XGBoost/LightGBM/CatBoost)을 각각 Optuna로 튜닝하고 best_params를
`docs/optuna_params.json`에 저장한다. (val 메트릭 최대화)
  - task별 메트릭: binary=ROC-AUC, multiclass=accuracy(argmax), regression=R².
  - 증분 저장: 각 (dataset,model) 직후 기록 → 중단돼도 진행분 보존.
  - 이미 캐시된 항목은 건너뜀 (재튜닝: --retune).
벤치마크(test 평가)는 `benchmark.py`가 이 JSON을 읽어 수행한다.

사용: python optimize.py [n_trials] [n_datasets_per_task] [--retune]
                          [--tasks binary,multiclass,regression]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tuning
import json, time
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score

from datasets import SUITES, CAT_INDEX
from tuning import (models_for, PARAMS_JSON, SEED, split_tvt,
                    inject_categorical, model_inputs)

optuna.logging.set_verbosity(optuna.logging.WARNING)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RETUNE   = "--retune" in sys.argv
NO_CAT   = "--no-categorical" in sys.argv   # 기본: 네이티브 범주 ON
N_TRIALS = int(ARGS[0]) if len(ARGS) > 0 else 30
N_DATA   = int(ARGS[1]) if len(ARGS) > 1 else 15
_tflag = [a for a in sys.argv if a.startswith("--tasks")]
TASKS = (_tflag[0].split("=")[1].split(",") if _tflag and "=" in _tflag[0]
         else ["binary", "multiclass", "regression"])


def load_cache():
    return json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else {}


def save_cache(cache):
    PARAMS_JSON.parent.mkdir(exist_ok=True)
    PARAMS_JSON.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def val_score(task, m, Xva, yva):
    """task별 validation 점수 (모두 '클수록 좋음').

    multiclass는 argmax **accuracy**로 튜닝한다 — predict()가 argmax를 쓰므로
    OvR-macro-AUC를 최대화하면 클래스 경계(threshold)는 좋아져도 argmax 정확도는
    안 맞을 수 있어, 헤드라인 acc가 약해졌다. binary는 그대로 AUC(이진 primary)."""
    if task == "binary":
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    if task == "multiclass":
        return accuracy_score(yva, m.predict(Xva))
    return r2_score(yva, m.predict(Xva))  # regression


def main():
    cache = load_cache()
    for task in TASKS:
        strat = task != "regression"
        data = SUITES[task](max_rows=10000)[:N_DATA]
        models = models_for(task)
        print(f"\n=== task: {task} ({len(data)} datasets) ===")
        for name, X, y in data:
            Xtr, Xva, _, ytr, yva, _ = split_tvt(X, y, stratify=strat)
            cat = [] if NO_CAT else CAT_INDEX.get(name, [])
            cards = {j: int(round(X[:, j].max())) + 1 for j in cat}
            line = f"  {name:20s} n={len(y):5d} d={X.shape[1]:4d} cat={len(cat):2d}  "
            for mname, (space, Model) in models.items():
                key = cache.get(name, {})
                if not RETUNE and key.get(mname) is not None and key.get("_task", "binary") == task:
                    line += f"{mname}=cached  "
                    continue
                Xtr_m = model_inputs(mname, Xtr, cat, cards)
                Xva_m = model_inputs(mname, Xva, cat, cards)
                t0 = time.perf_counter()
                def obj(t, _m=mname, _M=Model, _s=space, _Xt=Xtr_m, _Xv=Xva_m):
                    kw = _s(t); inject_categorical(_m, kw, cat)
                    m = _M(**kw); m.fit(_Xt, ytr)
                    return val_score(task, m, _Xv, yva)
                study = optuna.create_study(
                    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
                trials = 5 if mname in ("ObliqueTree", "ObliqueForest") else N_TRIALS
                study.optimize(obj, n_trials=trials, show_progress_bar=False)
                cache.setdefault(name, {})["_task"] = task
                cache[name][mname] = study.best_params
                save_cache(cache)                    # 증분 저장
                line += f"{mname}={study.best_value:.4f}({time.perf_counter()-t0:.0f}s)  "
            print(line)

    print(f"\n  best params → {PARAMS_JSON}")
    print(f"  이제 `python benchmark.py`로 test 평가.")


if __name__ == "__main__":
    main()
