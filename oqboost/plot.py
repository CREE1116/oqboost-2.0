"""oqboost.plot — OQBoost 네이티브 설명 시각화 (matplotlib).

shap 의존성 없이 OQBoost 고유 설명을 그린다. matplotlib는 lazy import이므로
이 모듈을 쓰지 않으면 설치하지 않아도 된다.

    import oqboost.plot as oqp
    oqp.plot_importance(model)         # 피처 중요도 (gain 또는 gain·|coef|)
    oqp.plot_interactions(model)       # 사선쌍 interaction 히트맵 (OQBoost 고유)
    oqp.plot_explanation(model, x)     # 한 표본의 가산적 기여 (워터폴)
    oqp.plot_explanation_summary(model, X)  # 다수 표본 beeswarm 요약
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted


def _lazy_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("oqboost.plot은 matplotlib이 필요합니다: "
                          "`pip install matplotlib`") from e
    return plt


def _n_features(model):
    check_is_fitted(model, "n_features_in_")
    return int(model.n_features_in_)


def _names(model, feature_names):
    d = _n_features(model)
    if feature_names is None:
        return [f"f{i}" for i in range(d)]
    if len(feature_names) != d:
        raise ValueError(f"feature_names 길이 {len(feature_names)} ≠ n_features {d}")
    return list(feature_names)


def _new_ax(ax, figsize):
    if ax is not None:
        return ax
    plt = _lazy_plt()
    _, ax = plt.subplots(figsize=figsize)
    return ax


def _is_multiclass(model):
    return bool(getattr(model, "_multiclass", False))


def _class_label(model, ci):
    cls = getattr(model, "classes_", None)
    return cls[ci] if cls is not None else ci


def _explain_2d(model, X, class_idx, default="mode"):
    """explain()을 (n, d)로 변환. 다중클래스면 class_idx 선택.

    default="mode": 가장 많이 예측된 클래스, "pred": 표본별 예측 클래스(단일 표본용).
    반환: (phi 2d, 선택된 클래스 라벨 또는 None)."""
    phi = np.asarray(model.explain(X), float)
    if phi.ndim == 2:                                  # 이진/회귀
        return phi, None
    K = phi.shape[1]                                   # (n, K, d)
    if class_idx is not None:
        ci = int(class_idx)
    elif default == "pred":
        pred = model.predict(X)
        classes = list(getattr(model, "classes_", range(K)))
        ci = classes.index(pred[0])
    else:                                              # 최빈 예측 클래스
        pred = model.predict(X)
        classes = list(getattr(model, "classes_", range(K)))
        vals, cnts = np.unique(pred, return_counts=True)
        ci = classes.index(vals[int(np.argmax(cnts))])
    return phi[:, ci, :], _class_label(model, ci)


def _imp_vector(model, method_name, class_idx):
    """importance 벡터. 다중클래스+class_idx면 그 클래스 부스터, 아니면 집계(평균)."""
    if class_idx is not None and _is_multiclass(model):
        b = model._boosters[int(class_idx)]
        v = np.asarray(getattr(b, method_name)(), float)
        s = v.sum()
        return (v / s if s > 0 else v), _class_label(model, int(class_idx))
    prop = {"coefficient_importances": "coefficient_importances_",
            "feature_importances": "feature_importances_",
            "interaction_importances": "interaction_importances_"}[method_name]
    return np.asarray(getattr(model, prop), float), None


def plot_importance(model, kind="coef", max_features=20, feature_names=None,
                    ax=None, color="#E05A2B", class_idx=None):
    """피처 중요도 가로 막대.

    kind="gain"  → feature_importances_ (Σ gain)
    kind="coef"  → coefficient_importances_ (Σ gain·|coef|, 방향 가중, 기본)
    다중클래스: class_idx 지정 시 그 클래스 부스터, 없으면 전 클래스 평균 집계.
    """
    if kind == "gain":
        imp, lab = _imp_vector(model, "feature_importances", class_idx)
        title = "Feature importance (Σ gain)"
    elif kind == "coef":
        imp, lab = _imp_vector(model, "coefficient_importances", class_idx)
        title = "Feature importance (Σ gain·|coef|)"
    else:
        raise ValueError("kind는 'gain' 또는 'coef'")
    if lab is not None:
        title += f"  ·  class {lab}"
    imp = np.asarray(imp, float)
    names = _names(model, feature_names)
    order = np.argsort(imp)[::-1][:max_features]
    order = order[::-1]  # 큰 값이 위로 오도록
    ax = _new_ax(ax, (7, max(3, 0.35 * len(order))))
    ax.barh(range(len(order)), imp[order], color=color, alpha=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("relative importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_interactions(model, feature_names=None, top=None, ax=None,
                      cmap="OrRd", annotate=None, class_idx=None):
    """사선쌍 interaction 히트맵 (OQBoost 고유). I_ij = Σ gain·|a|·|b|.

    top: 총 interaction 상위 N개 피처만 표시 (None=전체).
    annotate: 셀에 값 표기 (None=피처 ≤12면 자동).
    다중클래스: class_idx 지정 시 그 클래스 부스터, 없으면 전 클래스 평균 집계.
    """
    I, lab = _imp_vector(model, "interaction_importances", class_idx)
    I = np.asarray(I, float)
    I = I + I.T  # 상삼각 저장 → 대칭화
    names = _names(model, feature_names)
    d = I.shape[0]
    idx = np.arange(d)
    if top is not None and top < d:
        strength = I.sum(axis=1)
        idx = np.argsort(strength)[::-1][:top]
        idx = np.sort(idx)
        I = I[np.ix_(idx, idx)]
    sub = [names[i] for i in idx]
    if annotate is None:
        annotate = len(idx) <= 12
    ax = _new_ax(ax, (1.0 + 0.6 * len(idx), 1.0 + 0.6 * len(idx)))
    im = ax.imshow(I, cmap=cmap)
    ax.set_xticks(range(len(idx)), sub, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(idx)), sub, fontsize=8)
    ax.set_title("Pairwise interactions (Σ gain·|a|·|b|)"
                 + (f"  ·  class {lab}" if lab is not None else ""))
    if annotate:
        mx = I.max() or 1.0
        for a in range(len(idx)):
            for b in range(len(idx)):
                if I[a, b] > 0:
                    ax.text(b, a, f"{I[a, b]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if I[a, b] > 0.5 * mx else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_explanation(model, x, feature_names=None, max_features=12, ax=None,
                     class_idx=None):
    """한 표본의 가산적 피처 기여 (워터폴 스타일). 양수는 예측을 올린 피처, 음수는
    내린 피처. Σφ = 점수 − base. 다중클래스는 class_idx(기본=예측 클래스)의 OvR 점수
    기여를 그린다."""
    x = np.asarray(x, float).reshape(1, -1)
    phi2, lab = _explain_2d(model, x, class_idx, default="pred")
    phi = phi2[0]
    names = _names(model, feature_names)
    order = np.argsort(np.abs(phi))[::-1][:max_features]
    order = order[::-1]
    vals = phi[order]
    colors = ["#C0392B" if v < 0 else "#1E8449" for v in vals]
    ax = _new_ax(ax, (7, max(3, 0.4 * len(order))))
    ax.barh(range(len(order)), vals, color=colors, alpha=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("contribution to score (Σ = score − base)")
    ax.set_title("Why this prediction" + (f"  ·  class {lab}" if lab is not None else ""))
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_explanation_summary(model, X, feature_names=None, max_features=20,
                             ax=None, max_samples=2000, cmap="coolwarm",
                             class_idx=None):
    """다수 표본 설명 요약 (SHAP summary 유사 beeswarm). 피처별로 φ를 흩뿌리고
    점 색은 피처 값(낮음→높음). 피처는 평균 |φ|로 정렬. 다중클래스는 class_idx
    (기본=최빈 예측 클래스) 한 클래스의 OvR 기여를 그린다."""
    X = np.asarray(X, float)
    if X.shape[0] > max_samples:
        sel = np.random.RandomState(0).choice(X.shape[0], max_samples, False)
        X = X[sel]
    phi, lab = _explain_2d(model, X, class_idx, default="mode")
    names = _names(model, feature_names)
    imp = np.abs(phi).mean(axis=0)
    order = np.argsort(imp)[::-1][:max_features]
    order = order[::-1]
    ax = _new_ax(ax, (8, max(3, 0.4 * len(order))))
    for row, fi in enumerate(order):
        ph = phi[:, fi]
        xv = X[:, fi]
        rng = np.ptp(xv) or 1.0
        cval = (xv - xv.min()) / rng
        jitter = (np.random.RandomState(fi).rand(len(ph)) - 0.5) * 0.6
        sc = ax.scatter(ph, np.full_like(ph, row) + jitter, c=cval, cmap=cmap,
                        s=8, alpha=0.6, edgecolors="none")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("contribution φ (per sample)")
    ax.set_title("Explanation summary" + (f"  ·  class {lab}" if lab is not None else ""))
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("feature value", fontsize=8)
    cb.set_ticks([])
    return ax
