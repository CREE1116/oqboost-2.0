# OQBoost — Design Skeleton (v3)

**2D-oblique gradient boosting** (이진 분류). 정수코드 범주를 연속 oblique 경로로
다루는 GBDT. 이 문서는 C++ 포팅 전 확정한 아키텍처 뼈대다. 결정 근거는 실험
(`scripts/` 의 ablation들)과 문헌(아래 Sources).

> 용어: 과거 "categorical-aware oblique"로 출발했으나, ablation 결과 **범주 특별처리는
> net-negative**로 판명. 현재 정체성은 "범주를 ordinal 숫자축으로 흡수하는 2D-oblique GBDT".

---

## 0. 핵심 설계 결정 (확정)

| # | 결정 | 근거 |
|---|------|------|
| D1 | 한 분할 = **최대 2피처 oblique** (`a·u + b·v < t`). full d-dim 안 감. | 2D가 사선 이득 대부분 + 과적합/탐색비용 회피의 변곡점. 고차는 depth+boosting 합성. |
| D2 | 범주 = **정수코드 그대로 연속 oblique 경로**(`cat_encoding="continuous"` 기본). | ablation: 전용 인코딩(node/global/rank) 전부 as-cont보다 못함. 최저 DOF=최고 일반화. |
| D3 | 부스팅 = 표준 **2차 Newton (XGBoost식)**. 별도 oblique 부스팅 불필요. | 문헌상 oblique GBT 일관성 성립; 래퍼는 표준이면 충분. |
| D4 | 범주 전용 인코딩(ordered TS / smoothing / target-rate / rank) **전부 비기본**. | adaptive 인코딩일수록 DOF↑→과적합. "lower train AUC→higher test AUC" 단조. escape hatch로만 보존. |
| D5 | 쌍 탐색 = 기본 **전수(정확도 정석)**. `n_screen`은 **opt-in 속도 레버**. | SIS 3~14배 가속(d↑일수록↑)하나 marginal corr이 **순수 상호작용쌍을 놓침**(bank −0.019). 속도는 주로 python 문제 → C++ 벡터화면 전수도 빠름, SIS 불필요할 수 있음. |

---

## 1. 부스팅 루프 (meta-algorithm)

표준 2차 Newton boosting. `classifier.py` 구현됨.

```
init_score = logit(mean(y))
raw = init_score
for t in 1..n_estimators:
    p = sigmoid(raw)
    g = p - y                # 1st order gradient
    h = p * (1 - p)          # 2nd order hessian
    tree = ObliqueTree.fit(X, g, h)
    raw += learning_rate * tree.predict(X)
predict_proba = sigmoid(raw)
```

ordered boosting류는 D4 폐기와 함께 보류(범주 특별처리 안 하므로 동기 약함).

---

## 2. 노드 분할 탐색

각 노드에서 1D·2D 후보를 평가하고 gain 최대를 채택. (`node.py`)

### 2.1 후보 피처 — SIS 스크리닝 (D5)
```
gc   = g - mean(g)
corr[f] = | mean((x[:,f]-mean) * gc) / (std(x[:,f])*std(gc)) |   # 잔차와의 marginal 상관
feats   = top-m by corr[f]      # m = n_screen (None이면 전수)
```
1D 후보 = `feats` 각각, 2D 후보 = `feats` 내 쌍만 → `O(m²)`.
잔차 `g` 기준이라 노드마다 적응적.

**한계 (측정됨):** marginal 상관은 **순수 상호작용쌍**(각자 잔차 상관 ≈0, 함께만 유효)을
탈락시킴. bank/adult 같은 interaction-heavy 데이터서 m=8이 −0.008~−0.019 AUC 손실.
→ 기본 None(전수). 보완안: marginal top-m ∪ 랜덤쌍, 또는 2단계 interaction screen.

### 2.2 1D 분할
- 앵커 기반 임계 `x[f] < t`. (`_eval_1d`)
- 범주 전용 1D 경로 없음(연속과 동일 취급).

### 2.3 2D oblique 분할 (핵심)
```
for (fA, fB) in screened_pairs:
    grid  = anchor_grid(x[fA], x[fB])        # k×k 앵커 셀 G/H 집계
    seed  = bhc_partition(grid)              # 셀 이진 레이블 (방향 seed)
    coef  = Hweighted_LSQ(anchor_coords, seed.labels, seed.H)   # 2×2, O(1)
    proj  = coef[0]*x[fA] + coef[1]*x[fB]
    t,gain= threshold_scan(proj, g, h, λ)    # 분위수 스캔
    keep best by gain
```
방향은 BHC 이산 레이블을 앵커 좌표에 H-가중 LSQ로 연속화 → 매끈한 사선.
범주축도 정수코드 `x[f]` 그대로 들어감(별도 인코딩 없음).

---

## 3. 범주 처리 (D2/D4 — 특별처리 안 함)

**선언된 범주를 연속피처처럼 통과시킨다.** classifier가 `cat_encoding="continuous"`면
내부적으로 `cat_features`를 비우고 정수코드를 연속 경로로 보냄.

- 범주축 = ordinal 정수코드. oblique 선형결합에 숫자로 합류.
- 한 컷 = 코드 임계 → 코드순 contiguous 부분집합 가름.
- 비연속 그룹(`{0,3}|{1,2}` 등)은 **depth로 한 조각씩 + boosting 가산**으로 합성.
- 각 컷 저DOF(임계 1개) → 과적합 안 함. 이게 as-cont가 전용 인코딩 이긴 이유.

**Escape hatch (비기본, 고카디널리티 대비 실험용):**
`cat_encoding ∈ {node, global, rank}` — 각각 per-node leaf-weight / 전역 target-rate /
target-order. 현 데이터(저·중 카디널리티)선 모두 열세. **고카디널리티 unordered는 미검증
프론티어** — 필요 시 LightGBM식 gradient-정렬 집합분할 검토.

---

## 4. 정규화 노브

| 노브 | 역할 | 권장 |
|------|------|------|
| `reg_lambda` | 리프/이득 L2 | 1 (큰 데이터선 ↑ 약간 도움) |
| `max_depth` | 고차 상호작용 깊이 (2D 컷 스택 수) | 3~4 (depth=3가 큰 데이터 과적합↓) |
| `min_samples` | 리프 최소 샘플 | 10~30 |
| `n_anchors` | 앵커 해상도 | 6~12 (크면 과적합) |
| `n_screen` (m) | SIS 상위 피처 수 | √d~8 |
| `learning_rate`,`n_estimators` | 부스팅 합성 | 0.06 / 120 |
| `subsample` | 트리당 행 비율 (stochastic) | **0.7~0.8 (과적합 핵심 레버)** |
| `colsample` | 노드당 피처 비율 | **0.8** |
| `cat_encoding`, `ts_smoothing` | 범주 escape hatch | 기본값 유지 |

---

## 5. C++ 포팅 데이터 구조 (continuous-core, 단순화)

D2/D4로 범주 서브시스템 제거 → 노드가 연속 2D-oblique 단일 형태로 수렴.

```cpp
struct Node {
  bool   is_leaf;
  float  weight;            // leaf value
  uint8  split_type;        // LEAF | 1D_CONT | 2D_OBLIQUE
  int    fA, fB;            // feature idx (fB=-1 if 1D)
  float  coefA, coefB, bias;
  int    left, right;       // child indices (flat arena)
};
struct ObliqueTree { std::vector<Node> nodes; };       // arena, 포인터 없음
struct Booster     { std::vector<ObliqueTree> trees; float init_score, lr; };
```
- 노드 arena(flat vector) — 포인터 재귀 대신 인덱스. 캐시 친화.
- **predict = 벡터화 배치** (현 파이썬 샘플별 `predict_one` 루프 = 주 병목, 제거).
- 앵커 집계/LSQ = 수동 2×2 (O(1)). 범주 인코딩 필드 불필요.
- escape-hatch 인코딩 쓸 경우만 level→scalar 맵 추가(옵션 빌드).

---

## 6. 현재 상태 / 계획

1. ✅ 범주 oblique 합류 + lookup 폐기 (`node.py`).
2. ✅ 범주 인코딩 ablation → **continuous 기본 확정** (D2/D4).
3. ✅ SIS 스크리닝 구현·ablation (`n_screen`) — 3~14배 가속하나 상호작용쌍 손실. **기본 전수 유지, opt-in 레버로 보존.**
4. ✅ **C++ 포팅** (`cpp/oqboost_core.cpp`, pybind11). continuous-core + arena predict + 노드당 앵커 precompute + OpenMP 쌍 병렬(12코어). **python 대비 45~98x → 산업 GBDT의 1~6x 이내.** 정확도 python과 ±0.013(이산 컬럼 tie-sensitivity, 버그 아님).
5. ✅ 정확도·속도 2차: **히스토그램 임계** + **stochastic subsample/colsample**(과적합 해소).
6. ✅ 커널 최적화: **전역 사전 binning(히스토그램 트릭)** — 노드별 `fit_anchors` O(n log n) 정렬 제거. max_bins=16(coarse 그리드가 방향 seed 안정 + 속도). 결과: **XGB·LGBM 정확도·속도 양면 제압**(§7).
7. ⏭ 잔여(원하면): histogram subtraction(child=parent−sibling), float+SIMD, 고-d SIS opt-in.

---

## 7. 벤치 스냅샷 (`scripts/benchmark.py`, 히스토그램 binning, max_bins=16)

4모델 공정 비교 (real 5 + synthetic 2D 6), 동일 HP ne120/lr.06/depth4/sub.8/col.8.

| 모델 | 평균 AUC rank | 평균 학습초 |
|---|---|---|
| CatBoost | 1.73 | 0.041 |
| **OQBoost** | **2.27** | **0.196** |
| LightGBM | 2.64 | 0.240 |
| XGBoost | 3.36 | 0.211 |

**OQBoost가 XGBoost·LightGBM을 정확도·속도 둘 다 앞섬** (2위). CatBoost만 위
(oblivious tree 속도 + native 범주). **강점**: oblique 구조(Spiral·GaussQuantiles·bank·
XOR/Checkerboard서 XGB 참패 대조). **약점**: 범주 많은 real(german/adult), breast(d=30)
0.44s(O(d²) 쌍 — SIS opt-in 여지). 경계 시각화: `scripts/output/decision_boundary.png`.

---

## Sources
- *Consistency of the oblique decision tree and its boosting and random forest*, arXiv:2211.12653 — oblique GBT 일관성, CART보다 적은 트리, θ+s gain 최대화.
- *Optimizing High-Dimensional Oblique Splits*, arXiv:2503.14381 — sparse split > dense, 스크리닝/정규화 권장.
- CatBoost ordered target statistics — 검토 후 **폐기**(너무 비쌈 + per-node DOF 문제 못 고침).