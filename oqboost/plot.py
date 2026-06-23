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


def plot_importance(model, kind="coef", max_features=20, feature_names=None,
                    ax=None, color="#E05A2B"):
    """피처 중요도 가로 막대.

    kind="gain"  → feature_importances_ (Σ gain)
    kind="coef"  → coefficient_importances_ (Σ gain·|coef|, 방향 가중, 기본)
    """
    if kind == "gain":
        imp = np.asarray(model.feature_importances_, float)
        title = "Feature importance (Σ gain)"
    elif kind == "coef":
        imp = np.asarray(model.coefficient_importances_, float)
        title = "Feature importance (Σ gain·|coef|)"
    else:
        raise ValueError("kind는 'gain' 또는 'coef'")
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
                      cmap="OrRd", annotate=None):
    """사선쌍 interaction 히트맵 (OQBoost 고유). I_ij = Σ gain·|a|·|b|.

    top: 총 interaction 상위 N개 피처만 표시 (None=전체).
    annotate: 셀에 값 표기 (None=피처 ≤12면 자동).
    """
    I = np.asarray(model.interaction_importances_, float)
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
    ax.set_title("Pairwise interactions (Σ gain·|a|·|b|)")
    if annotate:
        mx = I.max() or 1.0
        for a in range(len(idx)):
            for b in range(len(idx)):
                if I[a, b] > 0:
                    ax.text(b, a, f"{I[a, b]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if I[a, b] > 0.5 * mx else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_explanation(model, x, feature_names=None, max_features=12, ax=None):
    """한 표본의 가산적 피처 기여 (워터폴 스타일). φ는 explain()에서 — 양수는
    예측을 올린 피처, 음수는 내린 피처. Σφ = 예측 − base."""
    x = np.asarray(x, float).reshape(1, -1)
    phi = np.asarray(model.explain(x), float)[0]
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
    ax.set_xlabel("contribution to prediction (Σ = pred − base)")
    ax.set_title("Why this prediction")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_explanation_summary(model, X, feature_names=None, max_features=20,
                             ax=None, max_samples=2000, cmap="coolwarm"):
    """다수 표본 설명 요약 (SHAP summary 유사 beeswarm). 피처별로 φ를 흩뿌리고
    점 색은 피처 값(낮음→높음). 피처는 평균 |φ|로 정렬."""
    X = np.asarray(X, float)
    if X.shape[0] > max_samples:
        sel = np.random.RandomState(0).choice(X.shape[0], max_samples, False)
        X = X[sel]
    phi = np.asarray(model.explain(X), float)
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
    ax.set_title("Explanation summary")
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("feature value", fontsize=8)
    cb.set_ticks([])
    return ax
