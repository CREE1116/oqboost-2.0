"""
benchmark.py — test 평가 (튜닝과 분리)

`optimize.py`가 만든 `docs/optuna_params.json`의 best_params를 읽어, 각 모델을
train+val로 재학습하고 held-out test ROC-AUC를 비교한다. 튜닝은 하지 않는다.

사용: python optimize.py 30 15   # 먼저 튜닝
      python benchmark.py        # 그다음 평가

출력: docs/benchmark.csv, docs/benchmark_optuna.png (README용)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasets import load_openml_suite
from tuning import MODELS, COLORS, PARAMS_JSON, build, split_tvt

N_DATA = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def main():
    if not PARAMS_JSON.exists():
        sys.exit(f"[!] {PARAMS_JSON} 없음. 먼저 `python optimize.py` 실행.")
    cache = json.loads(PARAMS_JSON.read_text())

    rows = []
    for name, X, y in load_openml_suite()[:N_DATA]:
        Xtr, Xva, Xtt, ytr, yva, ytt = split_tvt(X, y)
        Xtrv = np.vstack([Xtr, Xva]); ytrv = np.concatenate([ytr, yva])
        rec = {"dataset": name, "n": len(y), "d": X.shape[1]}
        for mname in MODELS:
            params = cache.get(name, {}).get(mname)
            if params is None:
                rec[mname] = np.nan; continue
            m = build(mname, params); m.fit(Xtrv, ytrv)
            rec[mname] = roc_auc_score(ytt, m.predict_proba(Xtt)[:, 1])
        print(f"  {name:18s} n={len(y):5d} d={X.shape[1]:3d}  " +
              "  ".join(f"{m}={rec[m]:.4f}" for m in MODELS))
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("dataset")
    out = PARAMS_JSON.parent
    df.to_csv(out / "benchmark.csv")
    auc = df[list(MODELS)]
    print("\n" + "=" * 88); print("Test ROC-AUC (Optuna-tuned, from optuna_params.json)"); print("=" * 88)
    print(auc.round(4).to_string())
    rank = auc.rank(axis=1, ascending=False).mean().sort_values()
    print("\nmean AUC rank (1=best):"); print(rank.round(2).to_string())
    print("\nwin count:"); print(auc.idxmax(axis=1).value_counts().to_string())
    _plot(auc, rank, out / "benchmark_optuna.png")
    print(f"\n  → {out}/benchmark.csv, benchmark_optuna.png")


def _plot(auc, rank, path):
    import matplotlib.pyplot as plt
    auc = auc.dropna(how="any")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 1]})
    ds = list(auc.index); x = np.arange(len(ds)); w = 0.2
    for i, m in enumerate(auc.columns):
        ax1.bar(x + (i-1.5)*w, auc[m].values, w, label=m, color=COLORS[m], alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(ds, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("Test ROC-AUC"); ax1.set_ylim(max(0.5, auc.values.min()-0.03), 1.005)
    ax1.set_title("Optuna-tuned test AUC per dataset"); ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, alpha=0.3); ax1.set_axisbelow(True)
    rc = rank.sort_values()
    ax2.barh(range(len(rc)), rc.values, color=[COLORS[m] for m in rc.index], alpha=0.9)
    ax2.set_yticks(range(len(rc))); ax2.set_yticklabels(rc.index)
    ax2.invert_yaxis(); ax2.set_xlabel("mean rank (lower=better)"); ax2.set_title("Mean AUC rank")
    for i, v in enumerate(rc.values): ax2.text(v+0.03, i, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()