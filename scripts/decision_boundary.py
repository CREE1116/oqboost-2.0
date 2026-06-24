"""
decision_boundary.py — 결정경계 비교 (OQBoost / XGBoost / LightGBM / CatBoost)
다양한 합성 2D 데이터셋에서 각 모델의 P(y=1) 경계를 나란히 그린다.
출력: docs/decision_boundary.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))         # scripts/ (datasets, models)
sys.path.insert(0, str(Path(__file__).parent.parent))  # 프로젝트 루트 (anchor_tree)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from datasets import SYNTH_2D
from models import make_models

STEP = 0.02
CMAP = "RdBu_r"


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
    out = Path(__file__).parent.parent / "docs"; out.mkdir(exist_ok=True)
    path = out / "decision_boundary.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  → {path}")


if __name__ == "__main__":
    main()
