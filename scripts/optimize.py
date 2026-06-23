"""
optimize.py — Optuna 하이퍼파라미터 튜닝 (벤치마크와 분리)

다양한 OpenML 이진 데이터셋에서 OQBoost/XGBoost/LightGBM/CatBoost를 각각 Optuna로
튜닝(val AUC 최대화)하고, best_params를 `docs/optuna_params.json`에 저장한다.
  - 증분 저장: 각 (dataset,model) 직후 기록 → 중단돼도 진행분 보존.
  - 이미 캐시된 항목은 건너뜀 (재튜닝: --retune).
벤치마크(test 평가)는 `benchmark.py`가 이 JSON을 읽어 수행한다.

사용: python optimize.py [n_trials] [n_datasets] [--retune]
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import optuna
from sklearn.metrics import roc_auc_score

from datasets import load_openml_suite
from tuning import MODELS, PARAMS_JSON, SEED, split_tvt

optuna.logging.set_verbosity(optuna.logging.WARNING)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RETUNE   = "--retune" in sys.argv
N_TRIALS = int(ARGS[0]) if len(ARGS) > 0 else 30
N_DATA   = int(ARGS[1]) if len(ARGS) > 1 else 12


def load_cache():
    return json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else {}

def save_cache(cache):
    PARAMS_JSON.parent.mkdir(exist_ok=True)
    PARAMS_JSON.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def main():
    cache = load_cache()
    data = load_openml_suite()[:N_DATA]
    for name, X, y in data:
        Xtr, Xva, _, ytr, yva, _ = split_tvt(X, y)
        line = f"  {name:18s} n={len(y):5d} d={X.shape[1]:3d}  "
        for mname, (space, Model) in MODELS.items():
            if not RETUNE and cache.get(name, {}).get(mname) is not None:
                line += f"{mname}=cached  "
                continue
            t0 = time.perf_counter()
            def obj(t):
                m = Model(**space(t)); m.fit(Xtr, ytr)
                return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
            study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
            cache.setdefault(name, {})[mname] = study.best_params
            save_cache(cache)                       # 증분 저장
            line += f"{mname}={study.best_value:.4f}({time.perf_counter()-t0:.0f}s)  "
        print(line)

    print(f"\n  best params → {PARAMS_JSON}")
    print(f"  이제 `python benchmark.py`로 test 평가.")


if __name__ == "__main__":
    main()
