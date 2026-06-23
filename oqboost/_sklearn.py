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
        boosters = state.pop("_boosters", None)
        state["_booster_bytes"] = booster.serialize() if booster is not None else None
        state["_boosters_bytes"] = ([b.serialize() for b in boosters]
                                    if boosters is not None else None)
        return state

    def __setstate__(self, state):
        bb = state.pop("_booster_bytes", None)
        bbs = state.pop("_boosters_bytes", None)
        self.__dict__.update(state)
        if bb is not None:
            self._booster = _core.Booster(); self._booster.deserialize(bb)
        if bbs is not None:
            self._boosters = []
            for s in bbs:
                b = _core.Booster(); b.deserialize(s); self._boosters.append(b)

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
    """2D-oblique GBDT 분류기. 이진=네이티브, 다중클래스=one-vs-rest."""

    def fit(self, X, y):
        X, y = _check_Xy(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        Xc = np.ascontiguousarray(X, dtype=float)
        if len(self.classes_) < 2:
            raise ValueError("클래스가 2개 미만입니다.")
        elif len(self.classes_) == 2:
            self._multiclass = False
            self._booster = self._make_booster(objective=0)
            self._booster.fit(Xc, (y == self.classes_[1]).astype(float))
        else:
            # one-vs-rest: 클래스마다 이진 부스터
            self._multiclass = True
            self._boosters = []
            for cls in self.classes_:
                b = self._make_booster(objective=0)
                b.fit(Xc, (y == cls).astype(float))
                self._boosters.append(b)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_multiclass")
        Xc = np.ascontiguousarray(_check_X(X), dtype=float)
        if not self._multiclass:
            return self._booster.predict_proba(Xc)
        # OvR: 각 부스터의 P(class k) → 행 정규화
        P = np.column_stack([b.predict_proba(Xc)[:, 1] for b in self._boosters])
        P = np.clip(P, 1e-12, None)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        P = self.predict_proba(X)
        if not self._multiclass:
            return self.classes_[(P[:, 1] >= 0.5).astype(int)]
        return self.classes_[P.argmax(axis=1)]

    @property
    def feature_importances_(self):
        check_is_fitted(self, "_multiclass")
        if not self._multiclass:
            return self._booster.feature_importances()
        # OvR: 부스터 평균
        fi = np.mean([b.feature_importances() for b in self._boosters], axis=0)
        return fi / fi.sum() if fi.sum() > 0 else fi


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
