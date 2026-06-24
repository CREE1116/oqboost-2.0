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
from catboost import CatBoostClassifier, CatBoostRegressor
from oqboost import OQBoostClassifier, OQBoostRegressor

SEED = 42
PARAMS_JSON = Path(__file__).parent.parent / "docs" / "optuna_params.json"
COLORS = {"OQBoost": "#E05A2B", "XGBoost": "#2980B9",
          "LightGBM": "#27AE60", "CatBoost": "#8E44AD"}


# ─── 모델별 search space (Optuna trial → kwargs) ─────────────────────────────
def oq_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 4),
                max_bins=t.suggest_int("max_bins", 8, 255),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample=t.suggest_float("colsample", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                n_screen = t.suggest_int("n_screen", 16, 16),
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
                bootstrap_type="Bernoulli",  # subsample 활성화를 위해 추가
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                verbose=False, random_seed=SEED, allow_writing_files=False)

MODELS = {
    "OQBoost":  (oq_params,  OQBoostClassifier),
    "XGBoost":  (xgb_params, xgb.XGBClassifier),
    "LightGBM": (lgb_params, lgb.LGBMClassifier),
    "CatBoost": (cat_params, CatBoostClassifier),
}


# ─── 회귀 search space (분류와 동형, n_screen=16 고정 — 실측상 최적 트레이드오프) ──
def oq_reg_params(t):
    # 회귀는 함수근사라 분류(1-4로 충분)보다 깊이를 더 원함 — house_16H/puma32H 등서
    # depth 4-6이 sweet spot(측정). 범위를 1-7로 확장.
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 7),
                max_bins=t.suggest_int("max_bins", 8, 255),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample=t.suggest_float("colsample", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                n_screen=t.suggest_int("n_screen", 16, 16),
                random_state=SEED)

def xgb_reg_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 3, 8),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                tree_method="hist", verbosity=0, random_state=SEED)

def lgb_reg_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 3, 8),
                num_leaves=t.suggest_int("num_leaves", 15, 127),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                subsample_freq=1, verbose=-1, random_state=SEED)

def cat_reg_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                depth=t.suggest_int("depth", 3, 8),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
                bootstrap_type="Bernoulli",  # subsample 활성화를 위해 추가
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                verbose=False, random_seed=SEED, allow_writing_files=False)

REG_MODELS = {
    "OQBoost":  (oq_reg_params,  OQBoostRegressor),
    "XGBoost":  (xgb_reg_params, xgb.XGBRegressor),
    "LightGBM": (lgb_reg_params, lgb.LGBMRegressor),
    "CatBoost": (cat_reg_params, CatBoostRegressor),
}

# task → 모델 레지스트리. binary·multiclass는 동일 분류기(다중클래스는 내부 처리).
REGISTRY = {"binary": MODELS, "multiclass": MODELS, "regression": REG_MODELS}


def models_for(task):
    return REGISTRY[task]


def inject_categorical(mname, kw, cat_idx):
    """모델별 네이티브 범주 처리 활성화 (kw 변형)."""
    if not cat_idx:
        return kw
    if mname == "OQBoost":
        kw["categorical_features"] = list(cat_idx)
    elif mname == "XGBoost":
        kw["enable_categorical"] = True
    elif mname == "CatBoost":
        kw["cat_features"] = list(cat_idx)
    # LightGBM: category dtype DataFrame을 fit서 자동 감지
    return kw


def model_inputs(mname, X, cat_idx, cards=None):
    """모델별 fit/predict 입력. 범주 인덱스 있으면 라이브러리별 형식으로 변환.

    OQBoost=numpy(인덱스로 처리), XGB/LGB=category dtype DataFrame,
    CatBoost=범주컬럼 문자열 DataFrame. `cards`(feature→레벨수)를 주면 category를
    range(card)로 고정 — train/test 분할 간 category↔code 매핑 정렬(미지 레벨 에러 방지)."""
    X = np.asarray(X, dtype=float)
    if not cat_idx or mname == "OQBoost":
        return np.ascontiguousarray(X)
    import pandas as pd
    df = pd.DataFrame(X)
    for j in cat_idx:
        codes = df[j].round().astype("int64")
        if mname == "CatBoost":
            df[j] = codes.astype(str)
        elif cards is not None and j in cards:
            df[j] = pd.Categorical(codes, categories=range(cards[j]))
        else:
            df[j] = codes.astype("category")
    return df


def build(mname, params, task="binary", cat_idx=None):
    """저장된 best_params(dict)로 모델 인스턴스 생성 (선택적 네이티브 범주)."""
    space, Model = REGISTRY[task][mname]
    kw = space(optuna.trial.FixedTrial(params))
    inject_categorical(mname, kw, cat_idx or [])
    return Model(**kw)


def split_tvt(X, y, seed=SEED, stratify=True):
    """train / val / test = 60 / 20 / 20. 분류는 stratified, 회귀는 무작위."""
    s1 = y if stratify else None
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=s1, random_state=seed)
    s2 = ytmp if stratify else None
    Xva, Xtt, yva, ytt = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=s2, random_state=seed)
    return Xtr, Xva, Xtt, ytr, yva, ytt