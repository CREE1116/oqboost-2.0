# OQBoost 2.0 — 2D-Oblique Gradient Boosting

## 한 줄 요약

OQBoost는 한 분할에서 **두 피처의 선형결합**(`a·u + b·v < t`)으로 **사선(oblique) 경계**를 직접 긋는 GBDT(Gradient Boosting Decision Tree)다. 축-정렬(axis-aligned) 트리가 계단식으로 근사하는 대각·상호작용 경계를 한 컷에 잡는다.

- **엔진**: C++ (`cpp/oqboost_core.cpp`, pybind11). 히스토그램 binning + OpenMP.
- **인터페이스**: scikit-learn. `OQBoostClassifier`(이진 분류) / `OQBoostRegressor`(회귀).
- **성능**: 동일 하이퍼파라미터에서 **XGBoost·LightGBM을 정확도·속도 둘 다 앞섬** (CatBoost만 위). §성능 참조.

> 이전 버전은 범주 전용 lookup·Voronoi 보간·적응적 |G|-앵커를 썼으나, ablation 결과
> 모두 net-negative/비핵심으로 판명되어 제거됐다. 현재는 정수코드 범주를 연속 oblique
> 경로로 흡수하는 단순·강력한 구조다. 설계 근거는 `docs/DESIGN.md`.

---

## 왜 사선인가 (동기)

표준 부스터(XGBoost, LightGBM)는 분할이 `x[f] < threshold` 형태의 **축-정렬**이다.
- 사선 경계(예: `x0·x1 > 0` XOR)를 표현하려면 계단식으로 여러 컷을 쌓아 경계가 거칠어진다.
- 실제로 벤치의 **XOR에서 XGBoost는 AUC 0.53**(무력)인데 OQBoost는 0.92, **Spiral**에서도 OQBoost가 가장 매끈한 경계를 그린다 (`scripts/output/decision_boundary.png`).

OQBoost는 한 노드에서 **두 피처를 동시에** 보고 그 평면에서 최적 사선을 찾아 이 격차를 직접 메운다.

---

## 어떻게 도는가

### 1. 부스팅 루프 (표준 2차 Newton)
```
init = logit(mean(y))            # 회귀: mean(y)
raw  = init
매 라운드 t:
    분류: p=sigmoid(raw); g=p-y; h=p(1-p)
    회귀: g=raw-y; h=1            # squared error
    tree = ObliqueTree.fit(X, g, h)
    raw += learning_rate * tree.predict(X)
```

### 2. 히스토그램 binning (1회, 커널 최적화)
학습 시작에 피처별 quantile 경계를 full-X에서 한 번 계산하고, 모든 샘플을 bin 인덱스로
미리 변환한다(`precompute_bins`). 이후 노드 분할 탐색이 **정렬 없이** bin 히스토그램
누적(O(n))으로 끝난다. `max_bins`(기본 16)는 **2D 그리드/방향 seed 해상도**만 제어
(coarse가 셀을 밀집시켜 방향 추정 안정 + 빠름). 임계 스캔은 별도 64-bin 히스토그램.

### 3. 노드 분할 — 1D vs 2D, gain 최대 채택
- **1D**: bin 히스토그램에서 최적 임계 `x[f] < t`.
- **2D oblique (핵심)** — 방향은 **H-가중 gradient 회귀**(기본, `fast_dir=1`):
  1. 피처쌍 `(fA,fB)`에서 per-sample Newton 타깃 `t=-g/h`를 두 raw 피처에 H-가중
     최소제곱 → 방향 `coef = (XᵀHX + λI)⁻¹ XᵀH t` (9-스칼라 1패스 + 2×2, O(1)).
  2. 투영 `proj = coef·x_raw` 위에서 히스토그램 임계 스캔 → `t`.
  - 분할: `coef[0]·x[fA] + coef[1]·x[fB] < t`.
  - legacy(`fast_dir=0`): bin 그리드 → BHC 이진 seed → 레이블 H-가중 LSQ. 약간 더
    정확할 수 있으나 그리드·정렬 비용으로 ~2배 느림.

각 노드 쌍 탐색은 **OpenMP 병렬**(피처쌍 단위).

### 4. 범주 처리 = 특별처리 없음
선언된 범주를 **정수코드 그대로 연속 oblique 경로**로 통과시킨다. 코드 임계가 코드순
부분집합을 가르고, 비연속 그룹은 depth+boosting 합성으로 표현. ablation 결과 전용
인코딩(lookup/leaf-weight/target-rate/rank)은 모두 과적합으로 net-negative였다.

---

## 설계 원칙 (왜 2D 고정인가)

한 컷은 **최대 2피처**. full d-차원 oblique 안 간다.
- 2D가 사선 이득 대부분 + 과적합/탐색비용 회피의 변곡점.
- 고차 상호작용은 **depth(2D 컷 스택) + boosting(가산)** 합성으로 도달.
- full d-dim은 분산 폭발 + `O(d^k)` 탐색 → 막다른 길 (ablation/문헌 확인).

과적합 억제는 **stochastic subsampling**(행 `subsample`, 노드 피처 `colsample`)이 핵심
레버 — train AUC~1.0을 깨 test AUC를 끌어올린다.

---

## 사용법

```python
from oqboost import OQBoostClassifier, OQBoostRegressor

clf = OQBoostClassifier(
    n_estimators=120, learning_rate=0.06, max_depth=4,
    max_bins=16, subsample=0.8, colsample=0.8, random_state=42,
)
clf.fit(X_train, y_train)            # 이진 분류
proba = clf.predict_proba(X_test)    # (n, 2)
pred  = clf.predict(X_test)

reg = OQBoostRegressor(n_estimators=120, learning_rate=0.06)
reg.fit(X_train, y_train)            # 회귀 (squared error)
yhat = reg.predict(X_test)
```

sklearn 호환: `get_params`/`set_params`/`clone`, Pipeline·GridSearchCV에 그대로.

빌드: `bash cpp/build.sh` → `oqboost/oqboost_core.*.so` 생성 (clang+libomp, cmake 불필요).

---

## 주요 하이퍼파라미터

| 파라미터 | 기본 | 의미 |
|----------|------|------|
| `n_estimators` | 120 | 부스팅 라운드 |
| `learning_rate` | 0.06 | 라운드별 기여 |
| `max_depth` | 4 | 고차 상호작용 깊이(2D 컷 스택) |
| `max_bins` | 16 | 그리드/방향 seed bin 수 (작게 유지) |
| `subsample` | 0.8 | 트리당 행 비율 (과적합 핵심 레버) |
| `colsample` | 0.8 | 노드당 피처 비율 |
| `reg_lambda` | 1.0 | L2 정규화 |
| `min_samples` | 10 | 리프 최소 샘플 |
| `n_screen` | -1 | SIS 상위 m피처만 쌍탐색 (-1=전수, opt-in 고-d 가속) |

---

## 성능 (5 real + 6 synthetic, held-out test)

`scripts/benchmark.py`, 4모델 동일 HP.

| 모델 | 평균 AUC rank | 평균 학습초 |
|---|---|---|
| CatBoost | 1.73 | 0.041 |
| **OQBoost** | **2.27** | **0.196** |
| LightGBM | 2.64 | 0.240 |
| XGBoost | 3.36 | 0.211 |

- **XGBoost·LightGBM을 정확도·속도 둘 다 앞섬** (2위). CatBoost만 우위.
- **강점**: oblique/상호작용 구조 — Spiral·GaussQuantiles·bank·diabetes 최상위, XOR/Checkerboard서 XGB 참패(0.53/0.74) 대비 0.92/0.93.
- **약점**: 범주 많은 real(german/adult), 고차원 d=30(breast) 단일 0.44s(O(d²) 쌍).

---

## 코드 맵

| 경로 | 역할 |
|------|------|
| `cpp/oqboost_core.cpp` | C++ 코어 — binning, Newton 부스팅, 1D/2D oblique, BHC seed, LSQ, 히스토그램 임계, SIS, arena predict, pybind11 |
| `cpp/build.sh` | clang 직접 빌드 (OpenMP) |
| `oqboost/_sklearn.py` | sklearn Classifier/Regressor 래퍼 |
| `oqboost/__init__.py` | 공개 API |
| `scripts/datasets.py` | 합성 2D 팩토리 + OpenML 로더 |
| `scripts/models.py` | 4모델 비교 팩토리 |
| `scripts/benchmark.py` | AUC/시간 벤치 |
| `scripts/decision_boundary.py` | 결정경계 시각화 |

---

## 한계

- **이진 분류 전용** (다중클래스 미지원). 회귀는 squared error.
- 2D 쌍 탐색 `O(d²)` — 고차원서 비용 (SIS opt-in 또는 향후 가지치기).
- 범주 native 처리(LightGBM/CatBoost식 집합분할) 없음 — 고카디널리티서 열세 가능.
