"""
tuning.py — optimize.py / benchmark.py 공유: 모델 search space + 생성 + 분할.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
def _suggest_cat_fallback(t, name, choices, default):
    if isinstance(t, optuna.trial.FixedTrial):
        if name in t.params:
            return t.suggest_categorical(name, choices)
        return default
    return t.suggest_categorical(name, choices)

from sklearn.model_selection import train_test_split
from obliquetree import Classifier as BaseOTClassifier, Regressor as BaseOTRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from oqboost import OQBoostClassifier, OQBoostRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class SklearnOTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        use_oblique: bool = True,
        max_depth: int = -1,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        categories = None,
        random_state = None,
        n_pair: int = 2,
        top_k = None,
        gamma: float = 1.0,
        max_iter: int = 100,
        relative_change: float = 0.001,
    ):
        self.use_oblique = use_oblique
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.categories = categories
        self.random_state = random_state
        self.n_pair = n_pair
        self.top_k = top_k
        self.gamma = gamma
        self.max_iter = max_iter
        self.relative_change = relative_change

    def fit(self, X, y, sample_weight=None):
        self.model_ = BaseOTClassifier(
            use_oblique=self.use_oblique,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            categories=self.categories,
            random_state=self.random_state,
            n_pair=self.n_pair,
            top_k=self.top_k,
            gamma=self.gamma,
            max_iter=self.max_iter,
            relative_change=self.relative_change,
        )
        self.model_.fit(X, y, sample_weight)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def apply(self, X):
        return self.model_.apply(X)

    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self, "model_") and hasattr(self.model_, "_categories"):
            state["_model_categories"] = self.model_._categories
        return state

    def __setstate__(self, state):
        categories = state.pop("_model_categories", None)
        self.__dict__.update(state)
        if categories is not None and hasattr(self, "model_"):
            self.model_._categories = categories

class SklearnOTRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        use_oblique: bool = True,
        max_depth: int = -1,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        categories = None,
        random_state = None,
        n_pair: int = 2,
        top_k = None,
        gamma: float = 1.0,
        max_iter: int = 100,
        relative_change: float = 0.001,
    ):
        self.use_oblique = use_oblique
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.categories = categories
        self.random_state = random_state
        self.n_pair = n_pair
        self.top_k = top_k
        self.gamma = gamma
        self.max_iter = max_iter
        self.relative_change = relative_change

    def fit(self, X, y, sample_weight=None):
        self.model_ = BaseOTRegressor(
            use_oblique=self.use_oblique,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            categories=self.categories,
            random_state=self.random_state,
            n_pair=self.n_pair,
            top_k=self.top_k,
            gamma=self.gamma,
            max_iter=self.max_iter,
            relative_change=self.relative_change,
        )
        self.model_.fit(X, y, sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def apply(self, X):
        return self.model_.apply(X)

    def __getstate__(self):
        state = self.__dict__.copy()
        if hasattr(self, "model_") and hasattr(self.model_, "_categories"):
            state["_model_categories"] = self.model_._categories
        return state

    def __setstate__(self, state):
        categories = state.pop("_model_categories", None)
        self.__dict__.update(state)
        if categories is not None and hasattr(self, "model_"):
            self.model_._categories = categories

class ObliqueForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        categories = None,
        random_state = None,
        n_pair: int = 2,
        top_k = None,
        gamma: float = 1.0,
        max_iter: int = 100,
        relative_change: float = 0.001,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.categories = categories
        self.random_state = random_state
        self.n_pair = n_pair
        self.top_k = top_k
        self.gamma = gamma
        self.max_iter = max_iter
        self.relative_change = relative_change
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        base = SklearnOTClassifier(
            use_oblique=True,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            categories=self.categories,
            random_state=self.random_state,
            n_pair=self.n_pair,
            top_k=self.top_k,
            gamma=self.gamma,
            max_iter=self.max_iter,
            relative_change=self.relative_change,
        )
        self.model_ = BaggingClassifier(
            estimator=base,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y, sample_weight)
        self.classes_ = self.model_.classes_
        self.n_classes_ = len(self.classes_)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def apply(self, X):
        return [est.apply(X) for est in self.model_.estimators_]

class ObliqueForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        categories = None,
        random_state = None,
        n_pair: int = 2,
        top_k = None,
        gamma: float = 1.0,
        max_iter: int = 100,
        relative_change: float = 0.001,
        n_jobs: int = 1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.categories = categories
        self.random_state = random_state
        self.n_pair = n_pair
        self.top_k = top_k
        self.gamma = gamma
        self.max_iter = max_iter
        self.relative_change = relative_change
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        base = SklearnOTRegressor(
            use_oblique=True,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            categories=self.categories,
            random_state=self.random_state,
            n_pair=self.n_pair,
            top_k=self.top_k,
            gamma=self.gamma,
            max_iter=self.max_iter,
            relative_change=self.relative_change,
        )
        self.model_ = BaggingRegressor(
            estimator=base,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y, sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def apply(self, X):
        return [est.apply(X) for est in self.model_.estimators_]

OTClassifier = SklearnOTClassifier
OTRegressor = SklearnOTRegressor
TRClassifier = ObliqueForestClassifier
TRRegressor = ObliqueForestRegressor

SEED = 42
PARAMS_JSON = Path(__file__).parent.parent / "docs" / "optuna_params.json"
COLORS = {"OQBoost": "#E05A2B", "XGBoost": "#2980B9",
          "LightGBM": "#27AE60", "CatBoost": "#8E44AD",
          "ObliqueTree": "#34495E", "ObliqueForest": "#D35400"}


# ─── 모델별 search space (Optuna trial → kwargs) ─────────────────────────────
def oq_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 8),
                max_bins=t.suggest_int("max_bins", 8, 255),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample=t.suggest_float("colsample", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                n_screen=16,
                fast_dir="full",  # 전수조사("full") 고정 — 정확도 우선
                random_state=SEED)

def oq_multiclass_params(t):
    """다중 클래스 전용 search space — fast_dir="full"(전수조사) 고정으로 정확도 우선."""
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 8),
                max_bins=t.suggest_int("max_bins", 8, 255),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample=t.suggest_float("colsample", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                n_screen=16,
                multiclass="joint",
                fast_dir="full",  # 멀티클래스도 전수조사("full") 고정
                random_state=SEED)

def xgb_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 8),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                tree_method="hist", eval_metric="logloss", verbosity=0, random_state=SEED)

def lgb_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 8),
                num_leaves=t.suggest_int("num_leaves", 15, 127),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                subsample_freq=1, verbose=-1, random_state=SEED)

def cat_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                depth=t.suggest_int("depth", 1, 8),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
                bootstrap_type="Bernoulli",  # subsample 활성화를 위해 추가
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                verbose=False, random_seed=SEED, allow_writing_files=False)


def ot_params(t):
    return dict(use_oblique=True,
                max_depth=t.suggest_int("max_depth", 3, 6),
                min_samples_split=t.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
                n_pair=2,
                random_state=SEED)


def tr_params(t):
    return dict(n_estimators=15,
                max_depth=t.suggest_int("max_depth", 3, 6),
                min_samples_split=t.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
                n_pair=2,
                n_jobs=1,
                random_state=SEED)


MODELS = {
    "OQBoost":  (oq_params,  OQBoostClassifier),
    "XGBoost":  (xgb_params, xgb.XGBClassifier),
    "LightGBM": (lgb_params, lgb.LGBMClassifier),
    "CatBoost": (cat_params, CatBoostClassifier),
    "ObliqueTree":  (ot_params,  OTClassifier),
    "ObliqueForest": (tr_params, TRClassifier),
}

# 다중 클래스 전용 레지스트리: OQBoost는 fast_dir="full"(전수조사) 고정 스페이스 사용
MC_MODELS = {
    "OQBoost":  (oq_multiclass_params, OQBoostClassifier),
    "XGBoost":  (xgb_params, xgb.XGBClassifier),
    "LightGBM": (lgb_params, lgb.LGBMClassifier),
    "CatBoost": (cat_params, CatBoostClassifier),
    "ObliqueTree":  (ot_params,  OTClassifier),
    "ObliqueForest": (tr_params, TRClassifier),
}


# ─── 회귀 search space (분류와 동형, n_screen=16 고정 — 실측상 최적 트레이드오프) ──
def oq_reg_params(t):
    # 회귀는 함수근사라 분류(1-4로 충분)보다 깊이를 더 원함 — house_16H/puma32H 등서
    # depth 4-6이 sweet spot(측정). 범위를 1-7로 확장.
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 8),
                max_bins=t.suggest_int("max_bins", 8, 255),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample=t.suggest_float("colsample", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                n_screen=t.suggest_int("n_screen", 16, 16),
                fast_dir="full",
                random_state=SEED)

def xgb_reg_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 8),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                tree_method="hist", verbosity=0, random_state=SEED)

def lgb_reg_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                max_depth=t.suggest_int("max_depth", 1, 8),
                num_leaves=t.suggest_int("num_leaves", 15, 127),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=t.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                subsample_freq=1, verbose=-1, random_state=SEED)

def cat_reg_params(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 60, 300),
                learning_rate=t.suggest_float("learning_rate", 0.02, 0.3, log=True),
                depth=t.suggest_int("depth", 1, 8),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
                bootstrap_type="Bernoulli",  # subsample 활성화를 위해 추가
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                verbose=False, random_seed=SEED, allow_writing_files=False)

REG_MODELS = {
    "OQBoost":  (oq_reg_params,  OQBoostRegressor),
    "XGBoost":  (xgb_reg_params, xgb.XGBRegressor),
    "LightGBM": (lgb_reg_params, lgb.LGBMRegressor),
    "CatBoost": (cat_reg_params, CatBoostRegressor),
    "ObliqueTree":  (ot_params,  OTRegressor),
    "ObliqueForest": (tr_params, TRRegressor),
}

# task → 모델 레지스트리. binary·multiclass는 동일 분류기(다중클래스는 내부 처리).
REGISTRY = {"binary": MODELS, "multiclass": MC_MODELS, "regression": REG_MODELS}


def models_for(task):
    mflag = [a for a in sys.argv if a.startswith("--models")]
    if mflag and "=" in mflag[0]:
        targets = [m.strip() for m in mflag[0].split("=")[1].split(",")]
        orig = REGISTRY[task]
        return {k: v for k, v in orig.items() if k in targets}
    return REGISTRY[task]


def inject_categorical(mname, kw, cat_idx):
    """모델별 네이티브 범주 처리 활성화 및 OpenMP 충돌 방지 단일 스레드 강제."""
    if mname == "XGBoost":
        kw["n_jobs"] = 1
    elif mname == "LightGBM":
        kw["n_jobs"] = 1
    elif mname == "CatBoost":
        kw["thread_count"] = 1
        kw["verbose"] = False
        kw["allow_writing_files"] = False

    if not cat_idx:
        return kw
    if mname == "OQBoost":
        kw["categorical_features"] = list(cat_idx)
    elif mname == "XGBoost":
        kw["enable_categorical"] = True
    elif mname == "CatBoost":
        kw["cat_features"] = list(cat_idx)
    elif mname in ("ObliqueTree", "ObliqueForest"):
        kw["categories"] = [int(x) for x in cat_idx]
    # LightGBM: category dtype DataFrame을 fit서 자동 감지
    return kw


def model_inputs(mname, X, cat_idx, cards=None):
    """모델별 fit/predict 입력. 범주 인덱스 있으면 라이브러리별 형식으로 변환.

    OQBoost=numpy(인덱스로 처리), XGB/LGB=category dtype DataFrame,
    CatBoost=범주컬럼 문자열 DataFrame. `cards`(feature→레벨수)를 주면 category를
    range(card)로 고정 — train/test 분할 간 category↔code 매핑 정렬(미지 레벨 에러 방지)."""
    X = np.asarray(X, dtype=float)
    if not cat_idx or mname in ("OQBoost", "ObliqueTree", "ObliqueForest"):
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