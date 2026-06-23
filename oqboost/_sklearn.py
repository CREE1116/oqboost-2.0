"""
_sklearn.py — OQBoost 2.0 scikit-learn 인터페이스 (C++ 백엔드 래핑)

OQBoostClassifier : 이진 분류 (logistic)
OQBoostRegressor  : 회귀 (squared error)

둘 다 C++ `oqboost_core.Booster`를 백엔드로 쓴다. 2D-oblique Newton GBDT.
"""
import inspect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from . import oqboost_core as _core

# NaN 허용 (C++ 백엔드가 결측치 native 처리). sklearn 버전별 인자명 호환.
_FINITE_KW = ("ensure_all_finite"
              if "ensure_all_finite" in inspect.signature(check_array).parameters
              else "force_all_finite")


def _check_X(X):
    return check_array(X, dtype=float, **{_FINITE_KW: "allow-nan"})


def _check_Xy(X, y, **kw):
    return check_X_y(X, y, dtype=float, **{_FINITE_KW: "allow-nan"}, **kw)


class _BaseOQBoost(BaseEstimator):
    """공통 파라미터 + Booster 생성."""

    def __init__(
        self,
        n_estimators: int = 120,
        learning_rate: float = 0.06,
        max_depth: int = 4,
        max_bins: int = 16,
        reg_lambda: float = 1.0,
        min_samples: int = 10,
        n_screen: int = -1,
        subsample: float = 0.8,
        colsample: float = 0.8,
        fast_dir: int = 1,   # H-가중 gradient 회귀 방향(기본). 0=BHC seed(legacy)
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.reg_lambda = reg_lambda
        self.min_samples = min_samples
        self.n_screen = n_screen
        self.subsample = subsample
        self.colsample = colsample
        self.fast_dir = fast_dir
        self.random_state = random_state

    # ── pickle 직렬화: C++ 부스터를 bytes로 변환/복원 ────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        booster = state.pop("_booster", None)
        state["_booster_bytes"] = booster.serialize() if booster is not None else None
        return state

    def __setstate__(self, state):
        bb = state.pop("_booster_bytes", None)
        self.__dict__.update(state)
        if bb is not None:
            self._booster = _core.Booster()
            self._booster.deserialize(bb)

    @property
    def feature_importances_(self):
        """피처별 정규화 누적 gain (sklearn 관례)."""
        check_is_fitted(self, "_booster")
        return self._booster.feature_importances()

    def _make_booster(self, objective: int):
        return _core.Booster(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, max_bins=self.max_bins,
            reg_lambda=self.reg_lambda, min_samples=self.min_samples,
            n_screen=self.n_screen, subsample=self.subsample,
            colsample=self.colsample, seed=int(self.random_state),
            objective=objective, fast_dir=self.fast_dir,
        )


class OQBoostClassifier(_BaseOQBoost, ClassifierMixin):
    """2D-oblique gradient-boosted oblique trees — 이진 분류기 (C++ 백엔드)."""

    def fit(self, X, y):
        X, y = _check_Xy(X, y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("OQBoostClassifier는 이진 분류만 지원합니다.")
        y01 = (y == self.classes_[1]).astype(float)
        self.n_features_in_ = X.shape[1]
        self._booster = self._make_booster(objective=0)
        self._booster.fit(np.ascontiguousarray(X, dtype=float), y01)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_booster")
        X = _check_X(X)
        return self._booster.predict_proba(np.ascontiguousarray(X, dtype=float))

    def predict(self, X):
        idx = (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.classes_[idx]


class OQBoostRegressor(_BaseOQBoost, RegressorMixin):
    """2D-oblique gradient-boosted oblique trees — 회귀기 (C++ 백엔드, squared error)."""

    def fit(self, X, y):
        X, y = _check_Xy(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self._booster = self._make_booster(objective=1)
        self._booster.fit(np.ascontiguousarray(X, dtype=float), y.astype(float))
        return self

    def predict(self, X):
        check_is_fitted(self, "_booster")
        X = _check_X(X)
        return self._booster.predict_raw(np.ascontiguousarray(X, dtype=float))
