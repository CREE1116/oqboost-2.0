"""explain_demo.py — OQBoost 네이티브 설명 시각화 데모 → docs/explainability.png

oqboost.plot의 4개 패널(중요도 / 사선쌍 interaction / 단일표본 워터폴 / beeswarm 요약)을
알려진 구조의 합성 데이터에 그려, README용 figure를 만든다.

    python scripts/explain_demo.py
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oqboost import OQBoostRegressor
import oqboost.plot as oqp

OUT = Path(__file__).parent.parent / "docs" / "explainability.png"

# 알려진 구조: age×income 곱셈 상호작용 + capital(+)·debt(-) 선형 + hours 약함 + noise.
rng = np.random.RandomState(1)
n = 3000
X = rng.standard_normal((n, 7))
y = (1.6 * X[:, 0] * X[:, 1] + 1.1 * X[:, 2] - 0.9 * X[:, 3]
     + 0.3 * X[:, 4] + rng.randn(n) * 0.3)
names = ["age", "income", "capital", "debt", "hours", "noise1", "noise2"]

r = OQBoostRegressor(n_estimators=150, learning_rate=0.06, random_state=0).fit(X, y)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
oqp.plot_importance(r, kind="coef", feature_names=names, ax=axes[0, 0])
oqp.plot_interactions(r, feature_names=names, ax=axes[0, 1])
oqp.plot_explanation(r, X[0], feature_names=names, ax=axes[1, 0])
oqp.plot_explanation_summary(r, X, feature_names=names, ax=axes[1, 1])
fig.suptitle("OQBoost native explanations  "
             "(true: 1.6·age·income + 1.1·capital − 0.9·debt + 0.3·hours)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"  → {OUT}")
