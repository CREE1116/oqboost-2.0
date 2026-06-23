"""
tuning.py — optimize.py / benchmark.py 공유: 모델 search space + 생성 + 분할.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from oqboost import OQBoostClassifier

SEED = 42
PARAMS_JSON = Path(__file__).parent.parent / "docs" / "optuna_params.json"
COLORS = {"OQBoost": "#E05A2B", "XGBoost": "#2980B9",
          "LightGBM": "#27AE60", "CatBoost": "#8E44AD"}


# ─── 모델별 search space (Optuna trial → kwargs) ─────────────────────────────
def oq_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 3, 6),
                max_bins=t.suggest_int("max_bins", 8, 32),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample=t.suggest_float("colsample", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                random_state=SEED)

def xgb_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 3, 8),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                tree_method="hist", eval_metric="logloss", verbosity=0, random_state=SEED)

def lgb_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 3, 8),
                num_leaves=t.suggest_int("num_leaves", 15, 127),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                subsample_freq=1, verbose=-1, random_state=SEED)

def cat_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                depth=t.suggest_int("depth", 3, 8),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                verbose=False, random_seed=SEED, allow_writing_files=False)

MODELS = {
    "OQBoost":  (oq_params,  OQBoostClassifier),
    "XGBoost":  (xgb_params, xgb.XGBClassifier),
    "LightGBM": (lgb_params, lgb.LGBMClassifier),
    "CatBoost": (cat_params, CatBoostClassifier),
}


def build(mname, params):
    """저장된 best_params(dict)로 모델 인스턴스 생성."""
    space, Model = MODELS[mname]
    return Model(**space(optuna.trial.FixedTrial(params)))


def split_tvt(X, y, seed=SEED):
    """train / val / test = 60 / 20 / 20 (stratified)."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    Xva, Xtt, yva, ytt = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)
    return Xtr, Xva, Xtt, ytr, yva, ytt
