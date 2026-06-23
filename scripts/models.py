"""
models.py — 비교 모델 팩토리 (공통 하이퍼파라미터)
AnchorTree(cpp) / XGBoost / LightGBM / CatBoost
"""
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from oqboost import OQBoostClassifier

# 색상 팔레트 (시각화 공용)
COLORS = {
    "OQBoost": "#E05A2B", "XGBoost": "#2980B9",
    "LightGBM": "#27AE60", "CatBoost": "#8E44AD",
}


def make_models(seed=42):
    """이름 → 새 분류기 인스턴스 딕셔너리."""
    return {
        "OQBoost": OQBoostClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=4,
            subsample=0.8, colsample=0.8, random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=4, reg_lambda=1.0,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="logloss", verbosity=0, random_state=seed),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=4, reg_lambda=1.0,
            subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
            verbose=-1, random_state=seed),
        "CatBoost": CatBoostClassifier(
            n_estimators=120, learning_rate=0.06, depth=4, l2_leaf_reg=1.0,
            verbose=False, random_seed=seed, allow_writing_files=False),
    }