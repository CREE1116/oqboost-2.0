"""
_sklearn.py — OQBoost 2.0 scikit-learn 인터페이스 (C++ 백엔드 래핑)

OQBoostClassifier : 이진 분류 (logistic)
OQBoostRegressor  : 회귀 (squared error)

둘 다 C++ `oqboost_core.Booster`를 백엔드로 쓴다. 2D-oblique Newton GBDT.
"""
import inspect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.class_weight import compute_sample_weight

from . import oqboost_core as _core

# threshold="..." → 최적화 메트릭. proba는 calibrated이므로 0.5가 기본이지만
# 불균형 데이터에서 balanced accuracy/F1은 다른 cut이 최적.
_THRESHOLD_METRICS = {
    "balanced": balanced_accuracy_score,
    "f1":       lambda yt, yp: f1_score(yt, yp, zero_division=0),
}

# 회귀 손실 → C++ loss 코드. squared=L2, huber=robust, quantile=pinball.
_LOSS = {"squared": 0, "huber": 1, "quantile": 2}

# NaN allowed (the C++ backend handles missing values natively); arg name varies
# across sklearn versions.
_FINITE_KW = ("ensure_all_finite"
              if "ensure_all_finite" in inspect.signature(check_array).parameters
              else "force_all_finite")

# validate_data (sklearn >=1.6) sets n_features_in_ / feature_names_in_ and checks
# feature-count consistency between fit and predict; fall back to the method form.
try:
    from sklearn.utils.validation import validate_data as _validate_data

    def _vd(est, X, y=None, reset=True, **kw):
        if y is None:
            return _validate_data(est, X, reset=reset, **kw)
        return _validate_data(est, X, y, reset=reset, **kw)
except ImportError:  # sklearn 1.3–1.5
    def _vd(est, X, y=None, reset=True, **kw):
        if y is None:
            return est._validate_data(X, reset=reset, **kw)
        return est._validate_data(X, y, reset=reset, **kw)


def _densify(X):
    # the backend needs a dense array and reads every feature (O(d^2) pair search),
    # so sparse input is accepted but densified.
    import scipy.sparse as sp
    return X.toarray() if sp.issparse(X) else X


def _check_X(est, X):
    return _densify(_vd(est, X, reset=False, dtype=float, accept_sparse="csr",
                        **{_FINITE_KW: "allow-nan"}))


def _check_Xy(est, X, y, reset=True, **kw):
    X, y = _vd(est, X, y, reset=reset, dtype=float, accept_sparse="csr",
               **{_FINITE_KW: "allow-nan"}, **kw)
    return _densify(X), y


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
        threshold="0.5",     # 0.5 | "balanced" | "f1" — 이진 결정 cut (분류기만)
        loss: str = "squared",   # 회귀 손실: squared | huber | quantile (회귀기만)
        alpha: float = 0.9,      # huber=delta 분위 / quantile=목표 분위 (회귀기만)
        clip: bool = False,      # 예측을 train 타깃 범위로 clamp (회귀기만)
        monotone_constraints=None,  # 피처별 단조 제약 리스트 -1/0/+1 (길이=n_features)
        categorical_features=None,  # 범주형 피처 인덱스/마스크 → 무손실 비닝
        class_weight=None,          # classifier only: None | "balanced" | dict -> sample weights
        max_lineage: int = 0,       # LOB: >0 inherits ancestor directions (composed). 0=classic 2D
        warm_start: bool = False,   # True + higher n_estimators -> add trees to the existing model
        n_iter_no_change=None,      # early stopping: stop after this many rounds w/o val improvement (None=off)
        validation_fraction: float = 0.1,  # held-out fraction for early-stopping monitoring
        tol: float = 1e-4,          # min val-deviance improvement to count as progress
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
        self.threshold = threshold
        self.loss = loss
        self.alpha = alpha
        self.clip = clip
        self.monotone_constraints = monotone_constraints
        self.categorical_features = categorical_features
        self.class_weight = class_weight
        self.max_lineage = max_lineage
        self.warm_start = warm_start
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.random_state = random_state

    # NaN is handled natively by the backend; tell sklearn it is intentional.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True   # NaN handled natively
        tags.input_tags.sparse = True      # sparse accepted (densified internally)
        return tags

    def _more_tags(self):  # sklearn < 1.6
        return {"allow_nan": True, "X_types": ["2darray", "sparse"]}

    # ── pickle serialization: C++ booster <-> bytes ─────────────────────────
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

    @property
    def coefficient_importances_(self):
        """계수 가중 importance: Σ gain·|coef| (정규화). oblique 방향 기여 반영."""
        check_is_fitted(self, "_booster")
        return self._booster.coefficient_importances()

    @property
    def interaction_importances_(self):
        """사선쌍 interaction 행렬 (d×d 상삼각, 정규화): Σ gain·|a|·|b|.

        OQBoost 고유 — 모든 분할이 피처쌍이라 학습된 상호작용이 트리에 그대로
        들어있다. `[i, j]`(i<j)가 피처 i,j의 상호작용 세기."""
        check_is_fitted(self, "_booster")
        return self._booster.interaction_importances()

    def explain(self, X):
        """표본별 피처 기여 (n, n_features). φ_i = Σ_경유분할 lr·gain·|coef_i|·dir.

        경유한 경로의 분할만 사용 → "왜 이 예측이 나왔는가"에 직접 답한다.
        양수는 예측을 끌어올린 피처, 음수는 끌어내린 피처."""
        check_is_fitted(self, "_booster")
        if int(self.max_lineage) > 0:
            raise NotImplementedError(
                "explain()은 max_lineage=0(기본 2D)만 지원 — LOB의 합성 dense 방향엔 "
                "경로-가산 귀속이 정의되지 않음.")
        Xc = np.ascontiguousarray(_check_X(self, X), dtype=float)
        return self._booster.explain(Xc)

    def _make_booster(self, objective: int):
        if self.loss not in _LOSS:
            raise ValueError(f"loss='{self.loss}' 미지원 ({' | '.join(_LOSS)})")
        mono = self._monotone_list()
        cat = self._categorical_list()
        return _core.Booster(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, max_bins=self.max_bins,
            reg_lambda=self.reg_lambda, min_samples=self.min_samples,
            n_screen=self.n_screen, subsample=self.subsample,
            colsample=self.colsample, seed=int(self.random_state),
            objective=objective, fast_dir=self.fast_dir,
            loss=_LOSS[self.loss], alpha=float(self.alpha), clip=int(bool(self.clip)),
            monotone=mono, categorical=cat, max_lineage=int(self.max_lineage),
        )

    def _categorical_list(self):
        """categorical_features → 길이 n_features의 0/1 마스크(없으면 빈 리스트).

        인덱스 리스트([2,5]), bool 마스크([F,F,T,...]), 또는 0/1 마스크 허용."""
        cf = self.categorical_features
        if cf is None:
            return []
        d = self.n_features_in_
        arr = np.asarray(cf)
        if arr.dtype == bool:
            if len(arr) != d:
                raise ValueError(f"categorical_features bool 마스크 길이 {len(arr)} ≠ {d}")
            return [int(v) for v in arr]
        out = [0] * d
        for i in arr:
            out[int(i)] = 1          # 인덱스 리스트로 해석
        return out

    def _monotone_list(self):
        """monotone_constraints → 길이 n_features의 int 리스트(없으면 빈 리스트).

        리스트/배열(-1/0/+1) 또는 {feat_idx: dir} dict 허용. 길이 검증."""
        mc = self.monotone_constraints
        if mc is None:
            return []
        d = self.n_features_in_
        if isinstance(mc, dict):
            out = [0] * d
            for k, v in mc.items():
                out[int(k)] = int(v)
            return out
        out = [int(v) for v in mc]
        if len(out) != d:
            raise ValueError(
                f"monotone_constraints 길이 {len(out)} ≠ n_features {d}")
        if any(v not in (-1, 0, 1) for v in out):
            raise ValueError("monotone_constraints 값은 -1/0/+1만 가능")
        return out

    @staticmethod
    def _sw(sample_weight, n):
        """sample_weight -> array for the C++ backend. None -> empty (unweighted)."""
        if sample_weight is None:
            return np.empty(0, dtype=float)
        sw = np.ascontiguousarray(np.asarray(sample_weight, dtype=float).ravel())
        if sw.shape[0] != n:
            raise ValueError(f"sample_weight length {sw.shape[0]} != n_samples {n}")
        if np.any(sw < 0):
            raise ValueError("sample_weight must be non-negative")
        if sw.shape[0] and sw.sum() == 0:
            raise ValueError("all sample_weight values are zero")
        return sw

    def _es_split(self, y, stratify):
        """(tr_idx, va_idx) for early stopping; (None, None) if off or warm-start."""
        if self.n_iter_no_change is None or self.warm_start:
            return None, None
        tr, va = train_test_split(
            np.arange(len(y)), test_size=self.validation_fraction,
            random_state=int(self.random_state),
            stratify=(y if stratify else None))
        return tr, va

    def _fit_booster(self, b, Xc, yt, sw, tr, va):
        """booster.fit with or without early stopping. Returns best_iteration (-1=none)."""
        if va is None:
            b.fit(Xc, yt, sw)
            return -1
        swt = sw[tr] if sw.size else sw
        b.fit(np.ascontiguousarray(Xc[tr]), yt[tr], swt,
              X_val=np.ascontiguousarray(Xc[va]), y_val=yt[va],
              es_patience=int(self.n_iter_no_change), es_tol=float(self.tol))
        return b.best_iteration()


class OQBoostClassifier(ClassifierMixin, _BaseOQBoost):
    """2D-oblique GBDT 분류기. 이진=네이티브, 다중클래스=one-vs-rest."""

    def fit(self, X, y, sample_weight=None):
        """Fit the classifier.

        Parameters
        ----------
        X, y : training data; y may be binary or multiclass (handled one-vs-rest).
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. They scale each sample's gradient and hessian, so a
            sample's influence on the loss is proportional to its weight. Note: this
            is **not exactly equivalent to repeating rows** — the histogram bin edges
            are computed unweighted, so weighted and replicated fits differ at a
            second-order level (the gap shrinks as ``max_bins`` grows).
        """
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        warm = self.warm_start and getattr(self, "classes_", None) is not None
        X, y = _check_Xy(self, X, y, reset=not warm)
        check_classification_targets(y)  # reject continuous regression targets
        Xc = np.ascontiguousarray(X, dtype=float)
        sw = self._sw(sample_weight, len(y))
        if self.class_weight is not None:  # fold class weights into the sample weights
            cw = compute_sample_weight(self.class_weight, y)
            sw = cw if sw.size == 0 else sw * cw
        # warm-start: add trees on the same data (n_estimators delta); keep threshold.
        if warm and X.shape[1] == self.n_features_in_:
            if not self._multiclass:
                extra = self.n_estimators - self._booster.n_trees()
                if extra > 0:
                    self._booster.fit_more(
                        Xc, (y == self.classes_[1]).astype(float), extra, sw)
            else:
                for cls, b in zip(self.classes_, self._boosters):
                    extra = self.n_estimators - b.n_trees()
                    if extra > 0:
                        b.fit_more(Xc, (y == cls).astype(float), extra, sw)
            return self
        if X.shape[0] < 2:
            raise ValueError(f"need at least 2 samples to fit, got n_samples={X.shape[0]}")
        self.classes_ = np.unique(y)
        # n_features_in_ / feature_names_in_ set by _check_Xy(reset=True)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least 2 classes")
        tr, va = self._es_split(y, stratify=True)  # early stopping split (or None)
        if len(self.classes_) == 2:
            self._multiclass = False
            ybin = (y == self.classes_[1]).astype(float)
            self.decision_threshold_ = self._fit_threshold(Xc, ybin)
            self._booster = self._make_booster(objective=0)
            self.best_iteration_ = self._fit_booster(self._booster, Xc, ybin, sw, tr, va)
        else:
            # one-vs-rest: one binary booster per class
            self._multiclass = True
            self.decision_threshold_ = 0.5  # multiclass uses argmax, no cut
            self._boosters = []
            bi = []
            for cls in self.classes_:
                b = self._make_booster(objective=0)
                bi.append(self._fit_booster(b, Xc, (y == cls).astype(float), sw, tr, va))
                self._boosters.append(b)
            self.best_iteration_ = bi if va is not None else -1
        return self

    def _fit_threshold(self, Xc, ybin):
        """이진 결정 cut 결정. float이면 그대로, 메트릭명이면 holdout서 최적화.

        proba는 calibrated이므로 0.5가 기본. 불균형 데이터에서 balanced
        accuracy/F1을 극대화하려면 cut을 옮겨야 함 → stratified holdout에서
        보조 부스터로 OOF proba를 얻어 최적 threshold 탐색(누수 없음).
        """
        try:
            return float(self.threshold)
        except (TypeError, ValueError):
            pass
        metric = _THRESHOLD_METRICS.get(self.threshold)
        if metric is None:
            raise ValueError(f"threshold='{self.threshold}' 미지원 "
                             f"(0.5 | {' | '.join(_THRESHOLD_METRICS)})")
        # 양/음 클래스 모두 최소 2개 있어야 stratify 가능
        if min(int(ybin.sum()), int((1 - ybin).sum())) < 2:
            return 0.5
        Xt, Xh, yt, yh = train_test_split(
            Xc, ybin, test_size=0.25, stratify=ybin,
            random_state=int(self.random_state))
        aux = self._make_booster(objective=0)
        aux.fit(np.ascontiguousarray(Xt), yt)
        ph = aux.predict_proba(np.ascontiguousarray(Xh))[:, 1]
        cands = np.unique(np.quantile(ph, np.linspace(0.02, 0.98, 49)))
        scores = [metric(yh, (ph >= t).astype(int)) for t in cands]
        return float(cands[int(np.argmax(scores))])

    def predict_proba(self, X):
        check_is_fitted(self, "_multiclass")
        Xc = np.ascontiguousarray(_check_X(self, X), dtype=float)
        if not self._multiclass:
            return self._booster.predict_proba(Xc)
        # OvR: 각 부스터의 P(class k) → 행 정규화
        P = np.column_stack([b.predict_proba(Xc)[:, 1] for b in self._boosters])
        P = np.clip(P, 1e-12, None)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        P = self.predict_proba(X)
        if not self._multiclass:
            t = getattr(self, "decision_threshold_", 0.5)
            return self.classes_[(P[:, 1] >= t).astype(int)]
        return self.classes_[P.argmax(axis=1)]

    @property
    def feature_importances_(self):
        check_is_fitted(self, "_multiclass")
        if not self._multiclass:
            return self._booster.feature_importances()
        # OvR: 부스터 평균
        fi = np.mean([b.feature_importances() for b in self._boosters], axis=0)
        return fi / fi.sum() if fi.sum() > 0 else fi

    def _avg_norm(self, attr):
        v = np.mean([getattr(b, attr)() for b in self._boosters], axis=0)
        s = v.sum()
        return v / s if s > 0 else v

    @property
    def coefficient_importances_(self):
        check_is_fitted(self, "_multiclass")
        if not self._multiclass:
            return self._booster.coefficient_importances()
        return self._avg_norm("coefficient_importances")

    @property
    def interaction_importances_(self):
        check_is_fitted(self, "_multiclass")
        if not self._multiclass:
            return self._booster.interaction_importances()
        return self._avg_norm("interaction_importances")

    def explain(self, X):
        """표본별 피처 기여.

        이진: (n, n_features) — class-1 logit 기여.
        다중클래스(OvR): (n, n_classes, n_features) — `[:, k, :]`가 class-k의
        one-vs-rest logit에 대한 가산적 기여(각 클래스 부스터의 explain). 클래스별로
        "왜 이 클래스 점수가 이렇게 나왔나"를 답한다."""
        check_is_fitted(self, "_multiclass")
        if int(self.max_lineage) > 0:
            raise NotImplementedError(
                "explain()은 max_lineage=0(기본 2D)만 지원 (LOB 합성 방향 미지원).")
        Xc = np.ascontiguousarray(_check_X(self, X), dtype=float)
        if not self._multiclass:
            return self._booster.explain(Xc)
        # OvR: 클래스별 부스터 explain → (n, n_classes, n_features)
        per = [b.explain(Xc) for b in self._boosters]  # 각 (n, d)
        return np.stack(per, axis=1)


class OQBoostRegressor(RegressorMixin, _BaseOQBoost):
    """2D-oblique gradient-boosted oblique trees — 회귀기 (C++ 백엔드, squared error)."""

    def fit(self, X, y, sample_weight=None):
        """Fit the regressor.

        Parameters
        ----------
        X, y : training data (numeric target).
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights scaling each sample's gradient and hessian. Not
            exactly equivalent to repeating rows — bin edges are computed unweighted
            (a second-order difference that shrinks as ``max_bins`` grows).
        """
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        warm = self.warm_start and getattr(self, "_booster", None) is not None
        X, y = _check_Xy(self, X, y, reset=not warm, y_numeric=True)
        Xc = np.ascontiguousarray(X, dtype=float)
        yf = y.astype(float)
        sw = self._sw(sample_weight, len(y))
        # warm-start: add trees on the same data (n_estimators delta).
        if warm and X.shape[1] == self.n_features_in_:
            extra = self.n_estimators - self._booster.n_trees()
            if extra > 0:
                self._booster.fit_more(Xc, yf, extra, sw)
            return self
        # n_features_in_ / feature_names_in_ set by _check_Xy(reset=True)
        self._booster = self._make_booster(objective=1)
        tr, va = self._es_split(yf, stratify=False)
        self.best_iteration_ = self._fit_booster(self._booster, Xc, yf, sw, tr, va)
        return self

    def predict(self, X):
        check_is_fitted(self, "_booster")
        X = _check_X(self, X)
        return self._booster.predict_raw(np.ascontiguousarray(X, dtype=float))
