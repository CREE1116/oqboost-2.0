"""
benchmark.py — test 평가 (튜닝과 분리)

`optimize.py`가 만든 `docs/optuna_params.json`의 best_params를 읽어 각 모델을
train+val로 재학습하고 held-out test에서 비교한다. 튜닝은 하지 않는다.

메트릭: ROC-AUC, accuracy, balanced accuracy, train time, inference time.

사용: python optimize.py 30 12   # 먼저 튜닝
      python benchmark.py        # 그다음 평가
출력: docs/benchmark.csv (long), docs/benchmark_optuna.png (README용)
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score)

from datasets import load_openml_suite
from tuning import MODELS, COLORS, PARAMS_JSON, build, split_tvt

N_DATA = int(sys.argv[1]) if len(sys.argv) > 1 else 12
HIGHER = {"auc": True, "acc": True, "bacc": True,
          "train_s": False, "infer_s": False}  # rank 방향


def _infer_time(m, X, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); m.predict_proba(X); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    if not PARAMS_JSON.exists():
        sys.exit(f"[!] {PARAMS_JSON} 없음. 먼저 `python optimize.py` 실행.")
    cache = json.loads(PARAMS_JSON.read_text())

    rows = []
    for name, X, y in load_openml_suite()[:N_DATA]:
        Xtr, Xva, Xtt, ytr, yva, ytt = split_tvt(X, y)
        Xtrv = np.vstack([Xtr, Xva]); ytrv = np.concatenate([ytr, yva])
        for mname in MODELS:
            params = cache.get(name, {}).get(mname)
            if params is None:
                continue
            m = build(mname, params)
            t0 = time.perf_counter(); m.fit(Xtrv, ytrv); train_s = time.perf_counter() - t0
            proba = m.predict_proba(Xtt)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({
                "dataset": name, "model": mname,
                "auc":  roc_auc_score(ytt, proba),
                "acc":  accuracy_score(ytt, pred),
                "bacc": balanced_accuracy_score(ytt, pred),
                "train_s": train_s,
                "infer_s": _infer_time(m, Xtt),
            })
        print(f"  done {name}")

    df = pd.DataFrame(rows)
    out = PARAMS_JSON.parent
    df.to_csv(out / "benchmark.csv", index=False)

    # ── 메트릭별 피벗 + 모델 평균/순위 ───────────────────────────────────────
    summary = {}
    for met in ["auc", "acc", "bacc", "train_s", "infer_s"]:
        piv = df.pivot(index="dataset", columns="model", values=met)[list(MODELS)]
        print("\n" + "=" * 78); print(f"{met}"); print("=" * 78)
        print(piv.round(4).to_string())
        summary[met] = piv.mean()

    sm = pd.DataFrame(summary)[["auc", "acc", "bacc", "train_s", "infer_s"]]
    print("\n" + "=" * 78); print("SUMMARY — mean across datasets"); print("=" * 78)
    print(sm.round(4).to_string())
    # AUC mean rank
    aucpiv = df.pivot(index="dataset", columns="model", values="auc")[list(MODELS)]
    rank = aucpiv.rank(axis=1, ascending=False).mean().sort_values()
    print("\nmean AUC rank (1=best):"); print(rank.round(2).to_string())

    _plot(aucpiv, rank, sm, out / "benchmark_optuna.png")
    print(f"\n  → {out}/benchmark.csv, benchmark_optuna.png")


def _plot(auc, rank, sm, path):
    import matplotlib.pyplot as plt
    auc = auc.dropna(how="any")
    fig, axes = plt.subplots(1, 3, figsize=(19, 5),
                             gridspec_kw={"width_ratios": [3, 1, 1.3]})
    ax1, ax2, ax3 = axes
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
    ax2.invert_yaxis(); ax2.set_xlabel("mean rank"); ax2.set_title("Mean AUC rank")
    for i, v in enumerate(rc.values): ax2.text(v+0.03, i, f"{v:.2f}", va="center", fontsize=9)

    # 평균 train/infer 속도 (log)
    models = list(sm.index); xm = np.arange(len(models))
    ax3.bar(xm-0.2, sm["train_s"].values, 0.4, label="train", color="#555", alpha=0.85)
    ax3.bar(xm+0.2, sm["infer_s"].values, 0.4, label="infer", color="#bbb", alpha=0.85)
    ax3.set_yscale("log"); ax3.set_xticks(xm); ax3.set_xticklabels(models, rotation=30, fontsize=8)
    ax3.set_ylabel("seconds (log)"); ax3.set_title("Mean train / inference time"); ax3.legend(fontsize=8)
    ax3.yaxis.grid(True, alpha=0.3); ax3.set_axisbelow(True)

    plt.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()