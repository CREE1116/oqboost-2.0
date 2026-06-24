"""
decision_boundary.py — 결정경계 비교 (OQBoost / XGBoost / LightGBM / CatBoost)
다양한 합성 2D 데이터셋에서 각 모델의 P(y=1) 경계를 나란히 그린다.
출력: docs/images/decision_boundary.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))         # scripts/ (datasets, models)
sys.path.insert(0, str(Path(__file__).parent.parent))  # 프로젝트 루트 (anchor_tree)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score, accuracy_score

from datasets import SYNTH_2D
from models import make_models

STEP = 0.02
CMAP = "RdBu_r"
MC_CMAP = ListedColormap(["#E05A2B", "#2980B9", "#27AE60"])  # 3-class


# ── 3-class 2D 합성 (멀티클래스 결정경계용) ──────────────────────────────────
def mc_blobs(n=900, seed=0):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n, centers=3, cluster_std=1.7, random_state=seed)
    return (X - X.mean(0)) / X.std(0), y


def mc_spiral3(n=900, seed=1):
    rng = np.random.default_rng(seed); m = n // 3; xs = []; ys = []
    for k in range(3):
        t = np.sqrt(rng.uniform(0, 1, m)) * 3 * np.pi
        ang = t + k * 2 * np.pi / 3
        xs.append(np.c_[t * np.cos(ang), t * np.sin(ang)] + rng.normal(0, 0.45, (m, 2)))
        ys.append(np.full(m, k))
    X = np.vstack(xs); y = np.concatenate(ys)
    return X / np.abs(X).max(), y


def mc_gauss3(n=900, seed=2):
    from sklearn.datasets import make_gaussian_quantiles
    X, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=3, random_state=seed)
    return (X - X.mean(0)) / X.std(0), y


SYNTH_MC = {"Blobs-3": mc_blobs, "Spiral-3": mc_spiral3, "GaussQuantiles-3": mc_gauss3}


def boundary_mc(ax, mdl, X, y, title):
    x0, x1 = X[:, 0], X[:, 1]
    xx, yy = np.meshgrid(np.arange(x0.min()-.3, x0.max()+.3, STEP),
                         np.arange(x1.min()-.3, x1.max()+.3, STEP))
    pred = mdl.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, pred, levels=[-.5, .5, 1.5, 2.5], cmap=MC_CMAP, alpha=0.45)
    ax.scatter(x0, x1, c=y, cmap=MC_CMAP, s=7, alpha=0.7, edgecolors="none")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def boundary(ax, mdl, X, y, title):
    x0, x1 = X[:, 0], X[:, 1]
    xx, yy = np.meshgrid(
        np.arange(x0.min()-.3, x0.max()+.3, STEP),
        np.arange(x1.min()-.3, x1.max()+.3, STEP))
    grid = np.c_[xx.ravel(), yy.ravel()]
    p = mdl.predict_proba(grid)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, p, levels=25, cmap=CMAP, alpha=0.85)
    ax.contour(xx, yy, p, levels=[0.5], colors="white", linewidths=1.6, linestyles="--")
    ax.scatter(x0, x1, c=y, cmap=CMAP, s=7, alpha=0.5, edgecolors="none")
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


def main():
    datasets = list(SYNTH_2D.items())
    model_names = list(make_models().keys())
    nrow, ncol = len(datasets), len(model_names)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6*ncol, 2.6*nrow))

    for r, (dname, fn) in enumerate(datasets):
        X, y = fn()
        for c, mname in enumerate(model_names):
            mdl = make_models()[mname]
            mdl.fit(X, y)
            auc = roc_auc_score(y, mdl.predict_proba(X)[:, 1])
            ax = axes[r, c]
            boundary(ax, mdl, X, y, f"{mname}\n{dname}  AUC={auc:.3f}")
            if c == 0:
                ax.set_ylabel(dname, fontsize=10, rotation=90)

    fig.suptitle("Decision Boundaries — OQBoost vs XGBoost / LightGBM / CatBoost",
                 fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout()
    out = Path(__file__).parent.parent / "docs" / "images"; out.mkdir(parents=True, exist_ok=True)
    path = out / "decision_boundary.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {path}")

    # ── 멀티클래스(3-class) 결정경계 ─────────────────────────────────────────
    mc = list(SYNTH_MC.items())
    fig, axes = plt.subplots(len(mc), ncol, figsize=(2.6*ncol, 2.6*len(mc)))
    for r, (dname, fn) in enumerate(mc):
        X, y = fn()
        for c, mname in enumerate(model_names):
            mdl = make_models()[mname]; mdl.fit(X, y)
            acc = accuracy_score(y, mdl.predict(X))
            boundary_mc(axes[r, c], mdl, X, y, f"{mname}\n{dname}  acc={acc:.3f}")
            if c == 0:
                axes[r, c].set_ylabel(dname, fontsize=10, rotation=90)
    fig.suptitle("Multiclass decision regions (3 classes) — OQBoost vs XGBoost / LightGBM / CatBoost",
                 fontsize=12, fontweight="bold", y=1.002)
    plt.tight_layout()
    path_mc = out / "decision_boundary_multiclass.png"
    fig.savefig(path_mc, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {path_mc}")


if __name__ == "__main__":
    main()
