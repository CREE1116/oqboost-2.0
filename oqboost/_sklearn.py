"""
_sklearn.py — OQBoost 2.0 scikit-learn 인터페이스 (C++ 백엔드 래핑)

OQBoostClassifier : 이진 분류 (logistic)
OQBoostRegressor  : 회귀 (squared error)

둘 다 C++ `oqboost_core.Booster`를 백엔드로 쓴다. 2D-oblique Newton GBDT.
"""
import inspect
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.class_weight import compute_sample_weight

from . import oqboost_core as _core
from . import _encoders as _enc

# threshold="..." → 최적화 메트릭. proba는 calibrated이므로 0.5가 기본이지만
# 불균형 데이터에서 balanced accuracy/F1은 다른 cut이 최적.
_THRESHOLD_METRICS = {
    "balanced": balanced_accuracy_score,
    "f1":       lambda yt, yp: f1_score(yt, yp, zero_division=0),
}

# 회귀 손실 → C++ loss 코드. squared=L2, huber=robust, quantile=pinball.
_LOSS = {"squared": 0, "huber": 1, "quantile": 2}

# 2D-pair search mode → C++ fast_dir code. "full"=all pairs (O(d²), accuracy),
# "fast"=Star anchor (feat0 × rest, O(d)). Legacy ints 1/2 still accepted.
_SEARCH = {"full": 1, "fast": 2}


def _fast_dir_code(fast_dir):
    if isinstance(fast_dir, str):
        try:
            return _SEARCH[fast_dir]
        except KeyError:
            raise ValueError(
                f"fast_dir={fast_dir!r} 미지원 (full | fast)") from None
    code = int(fast_dir)
    if code not in (1, 2):
        raise ValueError(f"fast_dir={fast_dir!r} 미지원 (full | fast, 또는 1 | 2)")
    return code

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
        fast_dir="full",     # 2D-pair search: "full"(all pairs, default) | "fast"(Star anchor)
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
        multiclass: str = "joint",
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
        self.multiclass = multiclass

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
        """Explain instance predictions."""
        check_is_fitted(self, "_booster")
        if int(self.max_lineage) > 0:
            raise NotImplementedError(
                "explain()은 max_lineage=0(기본 2D)만 지원 — LOB의 합성 dense 방향엔 "
                "경로-가산 귀속이 정의되지 않음.")
        Xc = np.ascontiguousarray(_check_X(self, X), dtype=float)
        cat = getattr(self, "_cat_idx", [])
        if cat:
            Xc = _enc.transform(Xc, cat, self._cat_enc)
        return self._booster.explain(Xc)

    def _make_booster(self, objective: int):
        if self.loss not in _LOSS:
            raise ValueError(f"loss='{self.loss}' 미지원 ({' | '.join(_LOSS)})")
        mono = self._monotone_list()
        return _core.Booster(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, max_bins=self.max_bins,
            reg_lambda=self.reg_lambda, min_samples=self.min_samples,
            n_screen=self.n_screen, subsample=self.subsample,
            colsample=self.colsample, seed=int(self.random_state),
            objective=objective, fast_dir=_fast_dir_code(self.fast_dir),
            loss=_LOSS[self.loss], alpha=float(self.alpha), clip=int(bool(self.clip)),
            monotone=mono, categorical=[], max_lineage=int(self.max_lineage),
        )

    def _cat_indices(self):
        cf = self.categorical_features
        if cf is None:
            return []
        arr = np.asarray(cf)
        if arr.dtype == bool:
            d = self.n_features_in_
            if len(arr) != d:
                raise ValueError(f"categorical_features bool mask length {len(arr)} != {d}")
            return [int(i) for i in np.flatnonzero(arr)]
        return [int(i) for i in arr]

    def _monotone_list(self):
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

    @staticmethod
    def _merge_class_weight(sw, class_weight, y_binary):
        cw = compute_sample_weight(class_weight, y_binary)
        return cw if sw.size == 0 else sw * cw

    def _es_split(self, y, stratify):
        if self.n_iter_no_change is None or self.warm_start:
            return None, None
        tr, va = train_test_split(
            np.arange(len(y)), test_size=self.validation_fraction,
            random_state=int(self.random_state),
            stratify=(y if stratify else None))
        return tr, va

    def _fit_booster(self, b, Xc, yt, sw, tr, va):
        if va is None:
            b.fit(Xc, yt, sw)
            return -1
        swt = sw[tr] if sw.size else sw
        b.fit(np.ascontiguousarray(Xc[tr]), yt[tr], swt,
              X_val=np.ascontiguousarray(Xc[va]), y_val=yt[va],
              es_patience=int(self.n_iter_no_change), es_tol=float(self.tol))
        return b.best_iteration()


class OQBoostClassifier(ClassifierMixin, _BaseOQBoost):
    """2D-oblique GBDT 분류기. 이진=네이티브, 다중클래스=one-vs-rest / joint."""

    def fit(self, X, y, sample_weight=None):
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        warm = self.warm_start and getattr(self, "classes_", None) is not None
        X, y = _check_Xy(self, X, y, reset=not warm)
        check_classification_targets(y)
        Xc = np.ascontiguousarray(X, dtype=float)

        sw_orig = self._sw(sample_weight, len(y))
        sw = sw_orig.copy() if sw_orig.size else sw_orig

        cat = self._cat_indices()

        if warm and X.shape[1] == self.n_features_in_:
            if not self._multiclass:
                sw_warm = sw_orig.copy() if sw_orig.size else sw_orig
                if self.class_weight is not None:
                    ybin_warm = (y == self.classes_[1]).astype(int)
                    sw_warm = self._merge_class_weight(sw_warm, self.class_weight, ybin_warm)
                extra = self.n_estimators - self._booster.n_trees()
                if extra > 0:
                    Xt = _enc.transform(Xc, cat, self._cat_enc) if cat else Xc
                    self._booster.fit_more(
                        Xt, (y == self.classes_[1]).astype(float), extra, sw_warm)
            elif getattr(self, "_joint_multiclass", False):
                sw_warm = sw_orig.copy() if sw_orig.size else sw_orig
                if self.class_weight is not None:
                    sw_warm = self._merge_class_weight(sw_warm, self.class_weight, y)
                extra = self.n_estimators - self._booster.n_trees()
                if extra > 0:
                    Xt = _enc.transform(Xc, cat, self._cat_enc) if cat else Xc
                    ymult = np.searchsorted(self.classes_, y).astype(float)
                    self._booster.fit_more(Xt, ymult, extra, sw_warm)
            else:
                for k, (cls, b) in enumerate(zip(self.classes_, self._boosters)):
                    extra = self.n_estimators - b.n_trees()
                    if extra > 0:
                        ybin_k = (y == cls).astype(int)
                        sw_k = sw_orig.copy() if sw_orig.size else sw_orig
                        if self.class_weight is not None:
                            sw_k = self._merge_class_weight(sw_k, self.class_weight, ybin_k)
                        Xt = _enc.transform(Xc, cat, self._cat_enc[k]) if cat else Xc
                        b.fit_more(Xt, ybin_k.astype(float), extra, sw_k)
            return self

        if X.shape[0] < 2:
            raise ValueError(f"need at least 2 samples to fit, got n_samples={X.shape[0]}")
        self.classes_ = np.unique(y)
        self._cat_idx = cat
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least 2 classes")
        
        # OvR 스케일링 복원을 위해 각 클래스의 원본 기저 확률(Priors) 미리 계산
        self._priors = np.array([np.mean(y == cls) for cls in self.classes_])
        
        tr, va = self._es_split(y, stratify=True)
        seed = int(self.random_state)

        if len(self.classes_) == 2:
            self._multiclass = False
            self._joint_multiclass = False
            ybin = (y == self.classes_[1]).astype(float)
            if self.class_weight is not None:
                sw = self._merge_class_weight(sw_orig, self.class_weight, ybin.astype(int))
            if cat:
                Xc, self._cat_enc = _enc.fit_transform(Xc, cat, ybin, True, seed=seed)
            else:
                self._cat_enc = None
            self.decision_threshold_ = self._fit_threshold(Xc, ybin)
            self._booster = self._make_booster(objective=0)
            self.best_iteration_ = self._fit_booster(self._booster, Xc, ybin, sw, tr, va)

        elif self.multiclass == "joint":
            self._multiclass = True
            self._joint_multiclass = True
            self.decision_threshold_ = 0.5
            ymult = np.searchsorted(self.classes_, y).astype(float)
            if self.class_weight is not None:
                sw = self._merge_class_weight(sw_orig, self.class_weight, y)
            if cat:
                Xc, self._cat_enc = _enc.fit_transform(Xc, cat, ymult, True, seed=seed)
            else:
                self._cat_enc = None
            self._booster = self._make_booster(objective=2)
            self.best_iteration_ = self._fit_booster(self._booster, Xc, ymult, sw, tr, va)

        else:
            self._multiclass = True
            self._joint_multiclass = False
            self.decision_threshold_ = 0.5

            if isinstance(self.threshold, str) and self.threshold != "0.5":
                warnings.warn(
                    f"threshold='{self.threshold}'은 multiclass OvR 모드에서 무시됩니다. "
                    "predict()는 argmax를 사용하므로 decision_threshold_가 적용되지 않습니다. "
                    "대신 predict_proba 단에서 Prior Correction 보정이 작동합니다.",
                    UserWarning, stacklevel=2,
                )

            self._boosters = []
            self._cat_enc = []
            bi = []
            for cls in self.classes_:
                ybin_k = (y == cls).astype(int)
                sw_k = sw_orig.copy() if sw_orig.size else sw_orig
                if self.class_weight is not None:
                    sw_k = self._merge_class_weight(sw_k, self.class_weight, ybin_k)

                ybin_kf = ybin_k.astype(float)
                if cat:
                    Xk, enck = _enc.fit_transform(
                        Xc, cat, ybin_kf, True, seed=seed, sample_weight=sw_k
                    )
                else:
                    Xk, enck = Xc, None
                self._cat_enc.append(enck)
                b = self._make_booster(objective=0)
                bi.append(self._fit_booster(b, Xk, ybin_kf, sw_k, tr, va))
                self._boosters.append(b)
            self.best_iteration_ = bi if va is not None else -1

        return self

    def _fit_threshold(self, Xc, ybin):
        try:
            return float(self.threshold)
        except (TypeError, ValueError):
            pass
        metric = _THRESHOLD_METRICS.get(self.threshold)
        if metric is None:
            raise ValueError(f"threshold='{self.threshold}' 미지원 "
                             f"(0.5 | {' | '.join(_THRESHOLD_METRICS)})")
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
        cat = getattr(self, "_cat_idx", [])
        
        if not self._multiclass or getattr(self, "_joint_multiclass", False):
            Xt = _enc.transform(Xc, cat, self._cat_enc) if cat else Xc
            return self._booster.predict_proba(Xt)
            
        # OvR 확률 예측: 베이지안 사전 확률 보정(Prior Correction) 적용 파트
        cols = []
        for k, b in enumerate(self._boosters):
            Xt = _enc.transform(Xc, cat, self._cat_enc[k]) if cat else Xc
            # b.predict_proba는 가중치로 인해 변형된 [1-p, p] 확률 반환
            pb = b.predict_proba(Xt)[:, 1]
            pb = np.clip(pb, 1e-15, 1.0 - 1e-15)
            
            # 1. 왜곡된 이진 결합 상태의 Log-odds(Logit) 추출
            logit_balanced = np.log(pb / (1.0 - pb))
            
            # 2. class_weight='balanced'가 켜져 왜곡이 발생한 경우 기저 우선순위 복원
            if self.class_weight is not None:
                pk = self._priors[k]
                if 0.0 < pk < 1.0:
                    # True Logit = Balanced Logit + log(pk / (1 - pk))
                    logit_true = logit_balanced + np.log(pk / (1.0 - pk))
                else:
                    logit_true = logit_balanced
            else:
                logit_true = logit_balanced
                
            # 3. 보정된 로짓을 바탕으로 이진 사후 확률 계산
            p_true = 1.0 / (1.0 + np.exp(-logit_true))
            cols.append(p_true)
            
        # 결합 및 정규화 수행
        P = np.column_stack(cols)
        P_sum = P.sum(axis=1, keepdims=True)
        P_sum = np.where(P_sum == 0, 1.0, P_sum)
        return P / P_sum

    def predict(self, X):
        P = self.predict_proba(X)
        if not self._multiclass:
            t = getattr(self, "decision_threshold_", 0.5)
            return self.classes_[(P[:, 1] >= t).astype(int)]
        return self.classes_[P.argmax(axis=1)]

    @property
    def feature_importances_(self):
        check_is_fitted(self, "_multiclass")
        if not self._multiclass or getattr(self, "_joint_multiclass", False):
            return self._booster.feature_importances()
        fi = np.mean([b.feature_importances() for b in self._boosters], axis=0)
        return fi / fi.sum() if fi.sum() > 0 else fi

    def _avg_norm(self, attr):
        v = np.mean([getattr(b, attr)() for b in self._boosters], axis=0)
        s = v.sum()
        return v / s if s > 0 else v

    @property
    def coefficient_importances_(self):
        check_is_fitted(self, "_multiclass")
        if not self._multiclass or getattr(self, "_joint_multiclass", False):
            return self._booster.coefficient_importances()
        return self._avg_norm("coefficient_importances")

    @property
    def interaction_importances_(self):
        check_is_fitted(self, "_multiclass")
        if not self._multiclass or getattr(self, "_joint_multiclass", False):
            return self._booster.interaction_importances()
        return self._avg_norm("interaction_importances")

    def explain(self, X):
        check_is_fitted(self, "_multiclass")
        if getattr(self, "_joint_multiclass", False):
            raise NotImplementedError(
                "explain()은 multiclass='joint' 모드를 지원하지 않습니다. multiclass='ovr'을 사용하세요.")
        if int(self.max_lineage) > 0:
            raise NotImplementedError(
                "explain()은 max_lineage=0(기본 2D)만 지원 (LOB 합성 방향 미지원).")
        Xc = np.ascontiguousarray(_check_X(self, X), dtype=float)
        cat = getattr(self, "_cat_idx", [])
        if not self._multiclass:
            Xt = _enc.transform(Xc, cat, self._cat_enc) if cat else Xc
            return self._booster.explain(Xt)
        per = []
        for k, b in enumerate(self._boosters):
            Xt = _enc.transform(Xc, cat, self._cat_enc[k]) if cat else Xc
            per.append(b.explain(Xt))
        return np.stack(per, axis=1)


class OQBoostRegressor(RegressorMixin, _BaseOQBoost):
    """2D-oblique gradient-boosted oblique trees — 회귀기 (C++ 백엔드, squared error)."""

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
        fast_dir="full",   # 회귀도 전수 조사("full")가 기본
        threshold="0.5",
        loss: str = "squared",
        alpha: float = 0.9,
        clip: bool = False,
        monotone_constraints=None,
        categorical_features=None,
        max_lineage: int = 0,
        warm_start: bool = False,
        n_iter_no_change=None,
        validation_fraction: float = 0.1,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_bins=max_bins,
            reg_lambda=reg_lambda,
            min_samples=min_samples,
            n_screen=n_screen,
            subsample=subsample,
            colsample=colsample,
            fast_dir=fast_dir,
            threshold=threshold,
            loss=loss,
            alpha=alpha,
            clip=clip,
            monotone_constraints=monotone_constraints,
            categorical_features=categorical_features,
            max_lineage=max_lineage,
            warm_start=warm_start,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            tol=tol,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None):
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        warm = self.warm_start and getattr(self, "_booster", None) is not None
        X, y = _check_Xy(self, X, y, reset=not warm, y_numeric=True)
        Xc = np.ascontiguousarray(X, dtype=float)
        yf = y.astype(float)
        sw = self._sw(sample_weight, len(y))
        cat = self._cat_indices()
        if warm and X.shape[1] == self.n_features_in_:
            extra = self.n_estimators - self._booster.n_trees()
            if extra > 0:
                Xt = _enc.transform(Xc, cat, self._cat_enc) if cat else Xc
                self._booster.fit_more(Xt, yf, extra, sw)
            return self
        self._cat_idx = cat
        if cat:
            Xc, self._cat_enc = _enc.fit_transform(Xc, cat, yf, False, seed=int(self.random_state))
        else:
            self._cat_enc = None
        self._booster = self._make_booster(objective=1)
        tr, va = self._es_split(yf, stratify=False)
        self.best_iteration_ = self._fit_booster(self._booster, Xc, yf, sw, tr, va)
        return self

    def predict(self, X):
        check_is_fitted(self, "_booster")
        Xc = np.ascontiguousarray(_check_X(self, X), dtype=float)
        cat = getattr(self, "_cat_idx", [])
        if cat:
            Xc = _enc.transform(Xc, cat, self._cat_enc)
        return self._booster.predict_raw(Xc)