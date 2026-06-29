"""
models.py — 비교 모델 팩토리 (공통 하이퍼파라미터)
AnchorTree(cpp) / XGBoost / LightGBM / CatBoost
"""
import warnings
warnings.filterwarnings("ignore")

from tuning import SklearnOTClassifier, ObliqueForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from oqboost import OQBoostClassifier

# 색상 팔레트 (시각화 공용)
COLORS = {
    "OQBoost": "#E05A2B", "XGBoost": "#2980B9",
    "LightGBM": "#27AE60", "CatBoost": "#8E44AD",
    "ObliqueTree": "#34495E", "ObliqueForest": "#D35400"
}


def make_models(seed=42):
    """이름 → 새 분류기 인스턴스 딕셔너리."""
    return {
        "OQBoost": OQBoostClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=4,
            subsample=1.0, colsample=1.0, random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=4, reg_lambda=1.0,
            subsample=1.0, colsample_bytree=1.0, tree_method="hist",
            eval_metric="logloss", verbosity=0, random_state=seed, n_jobs=1),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=4, reg_lambda=1.0,
            subsample=1.0, colsample_bytree=1.0,
            verbose=-1, random_state=seed, n_jobs=1),
        "CatBoost": CatBoostClassifier(
            n_estimators=120, learning_rate=0.06, depth=4, l2_leaf_reg=1.0,
            verbose=False, random_seed=seed, allow_writing_files=False),
        "ObliqueTree": SklearnOTClassifier(
            use_oblique=True, max_depth=4, random_state=seed),
        "ObliqueForest": ObliqueForestClassifier(
            n_estimators=15, max_depth=4, random_state=seed, n_jobs=1),
    }