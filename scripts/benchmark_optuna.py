"""
benchmark_optuna.py — Optuna 일괄 튜닝 벤치마크 (best-params JSON 캐싱)

다양한 OpenML 이진 데이터셋에서 OQBoost / XGBoost / LightGBM / CatBoost를
Optuna로 동일 예산 튜닝(val AUC 최대화) → test AUC 비교.

튜닝 결과는 `docs/optuna_params.json`에 (dataset → model → best_params)로 저장된다.
  - 캐시가 있으면 재튜닝 없이 그대로 반영 (재현/빠른 재실행).
  - 증분 저장: 각 (dataset,model) 튜닝 직후 기록 → 중단돼도 진행분 보존.
  - 재튜닝 강제: `--retune`.

사용: python benchmark_optuna.py [n_trials] [n_datasets] [--retune]
"""
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from oqboost import OQBoostClassifier
from datasets import load_openml_suite

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
RETUNE   = "--retune" in sys.argv
N_TRIALS = int(ARGS[0]) if len(ARGS) > 0 else 30
N_DATA   = int(ARGS[1]) if len(ARGS) > 1 else 12
PARAMS_JSON = Path(__file__).parent.parent / "docs" / "optuna_params.json"
COLORS = {"OQBoost": "#E05A2B", "XGBoost": "#2980B9", "LightGBM": "#27AE60", "CatBoost": "#8E44AD"}


# ─── 모델별 search space + 생성자 ────────────────────────────────────────────
def oq_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators",60,300),
                learning_rate=t.suggest_float("learning_rate",0.02,0.3,log=True),
                max_depth=t.suggest_int("max_depth",3,6),
                max_bins=t.suggest_int("max_bins",8,32),
                subsample=t.suggest_float("subsample",0.6,1.0),
                colsample=t.suggest_float("colsample",0.6,1.0),
                reg_lambda=t.suggest_float("reg_lambda",0.1,5.0,log=True),
                random_state=SEED)

def xgb_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators",60,300),
                learning_rate=t.suggest_float("learning_rate",0.02,0.3,log=True),
                max_depth=t.suggest_int("max_depth",3,8),
                subsample=t.suggest_float("subsample",0.6,1.0),
                colsample_bytree=t.suggest_float("colsample_bytree",0.6,1.0),
                reg_lambda=t.suggest_float("reg_lambda",0.1,5.0,log=True),
                min_child_weight=t.suggest_int("min_child_weight",1,10),
                tree_method="hist", eval_metric="logloss", verbosity=0, random_state=SEED)

def lgb_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators",60,300),
                learning_rate=t.suggest_float("learning_rate",0.02,0.3,log=True),
                max_depth=t.suggest_int("max_depth",3,8),
                num_leaves=t.suggest_int("num_leaves",15,127),
                subsample=t.suggest_float("subsample",0.6,1.0),
                colsample_bytree=t.suggest_float("colsample_bytree",0.6,1.0),
                reg_lambda=t.suggest_float("reg_lambda",0.1,5.0,log=True),
                subsample_freq=1, verbose=-1, random_state=SEED)

def cat_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators",60,300),
                learning_rate=t.suggest_float("learning_rate",0.02,0.3,log=True),
                depth=t.suggest_int("depth",3,8),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg",0.5,10.0,log=True),
                subsample=t.suggest_float("subsample",0.6,1.0),
                verbose=False, random_seed=SEED, allow_writing_files=False)

MODELS = {
    "OQBoost":  (oq_params,  OQBoostClassifier),
    "XGBoost":  (xgb_params, xgb.XGBClassifier),
    "LightGBM": (lgb_params, lgb.LGBMClassifier),
    "CatBoost": (cat_params, CatBoostClassifier),
}


# ─── JSON 캐시 ───────────────────────────────────────────────────────────────
def load_cache():
    if PARAMS_JSON.exists():
        return json.loads(PARAMS_JSON.read_text())
    return {}

def save_cache(cache):
    PARAMS_JSON.parent.mkdir(exist_ok=True)
    PARAMS_JSON.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def get_best_params(make_params, Model, dataset, mname, Xtr, ytr, Xva, yva, cache):
    """캐시에 있으면 반환, 없거나 --retune이면 튜닝 후 캐시에 증분 저장."""
    if not RETUNE and cache.get(dataset, {}).get(mname) is not None:
        return cache[dataset][mname]
    def obj(t):
        m = Model(**make_params(t)); m.fit(Xtr, ytr)
        return roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    cache.setdefault(dataset, {})[mname] = study.best_params
    save_cache(cache)                       # 증분 저장
    return study.best_params


def main():
    cache = load_cache()
    data = load_openml_suite()[:N_DATA]
    rows = []
    for name, X, y in data:
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
        Xva, Xtt, yva, ytt = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=SEED)
        Xtrv = np.vstack([Xtr, Xva]); ytrv = np.concatenate([ytr, yva])
        rec = {"dataset": name, "n": len(y), "d": X.shape[1]}
        t0 = time.perf_counter()
        for mname, (mp, Model) in MODELS.items():
            bp = get_best_params(mp, Model, name, mname, Xtr, ytr, Xva, yva, cache)
            m = Model(**mp(optuna.trial.FixedTrial(bp)))
            m.fit(Xtrv, ytrv)
            rec[mname] = roc_auc_score(ytt, m.predict_proba(Xtt)[:, 1])
        print(f"  {name:18s} n={len(y):5d} d={X.shape[1]:3d}  " +
              "  ".join(f"{m}={rec[m]:.4f}" for m in MODELS) +
              f"   ({time.perf_counter()-t0:.0f}s)")
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("dataset")
    out = PARAMS_JSON.parent
    df.to_csv(out / "benchmark_optuna.csv")
    auc = df[list(MODELS)]
    print("\n" + "=" * 88); print(f"Test ROC-AUC (Optuna {N_TRIALS} trials/model, cached → {PARAMS_JSON.name})"); print("=" * 88)
    print(auc.round(4).to_string())
    rank = auc.rank(axis=1, ascending=False).mean().sort_values()
    print("\nmean AUC rank (1=best):"); print(rank.round(2).to_string())
    print("\nwin count:"); print(auc.idxmax(axis=1).value_counts().to_string())
    _plot(auc, rank, out / "benchmark_optuna.png")
    print(f"\n  → {out}/benchmark_optuna.{{csv,png}}, {PARAMS_JSON.name}")


def _plot(auc, rank, path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 1]})
    ds = list(auc.index); x = np.arange(len(ds)); w = 0.2
    for i, m in enumerate(auc.columns):
        ax1.bar(x + (i-1.5)*w, auc[m].values, w, label=m, color=COLORS[m], alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(ds, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Test ROC-AUC"); ax1.set_ylim(max(0.5, auc.values.min()-0.03), 1.005)
    ax1.set_title("Optuna-tuned test AUC per dataset"); ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3); ax1.set_axisbelow(True)
    rc = rank.sort_values()
    ax2.barh(range(len(rc)), rc.values, color=[COLORS[m] for m in rc.index], alpha=0.9)
    ax2.set_yticks(range(len(rc))); ax2.set_yticklabels(rc.index)
    ax2.invert_yaxis(); ax2.set_xlabel("mean rank (lower=better)"); ax2.set_title("Mean AUC rank")
    for i, v in enumerate(rc.values): ax2.text(v+0.03, i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()