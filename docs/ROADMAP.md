# OQBoost 2.0 — Feature Roadmap

결과론적 목표(차차 업데이트). 난이도·의존성 고려한 제안 순서.

## 현재 (구현됨)
- 2D-oblique GBDT, 히스토그램 binning, OpenMP 쌍병렬, fast_dir(gradient 회귀 방향, 기본).
- 이진 분류 + 회귀, scikit-learn API, **native NaN(결측치)**.
- 다중플랫폼 wheel CI(cibuildwheel), Optuna 벤치(AUC/acc/bacc/train·infer time).

## 로드맵

| # | 기능 | 난이도 | 메모 |
|---|------|--------|------|
| ☑ | **Model serialization** | 낮음 | ✅ 완료. C++ `serialize/deserialize`(Node POD memcpy) + 래퍼 `__getstate__/__setstate__` → pickle·joblib 호환. |
| ☑ | **Feature importance** | 낮음 | ✅ 완료. 채택 분할 gain 피처별 누적 → `feature_importances_`(정규화), pickle 보존. |
| ☑ | **분류 threshold 튜닝** | 낮음 | ✅ 완료. `threshold="balanced"/"f1"` → holdout서 cut 최적화(`decision_threshold_`). proba는 calibrated(mean≈base rate)이나 불균형서 0.5 cut은 bacc 붕괴 → 옮겨야 함. 기본 0.5 유지. |
| ☑ | **Regression 안정화** | 낮음 | ✅ 완료. `loss="huber"/"quantile"`(robust 손실), `clip`(예측 train 범위 clamp), huber/quantile은 init=median 자동. quantile은 leaf-value line-search(분위수 재계산). 이상치 train서 huber/median MAE ≪ squared. 한계: 극단 분위(α≈0.9) interval은 shallow tree서 안쪽 편향(median은 정확). |
| ☑ | **Thread scaling** | 중 | ✅ 완료. 작업량(쌍수×표본수) 임계 이하 노드는 serial 폴백(`if` clause)으로 fork-join 오버헤드 제거 — small-data 8스레드 회귀(0.105→0.081s) 해소. gradient/raw/predict n-루프도 임계 가드. big-data 4코어 ~2.8x(70%) 유지, 스레드 수 무관 bit-identical. |
| ☑ | **Monotonic constraints** | 중 | ✅ 완료. `monotone_constraints`(-1/0/+1, list 또는 dict). 풀 oblique 단조: 고정 타피처 직선 위 사선분할=단일 threshold라 axis 기법(중점 값-경계 전파 + leaf clamp)이 이식됨. 쌍 둘다 제약시 `sign(coefA)·mA==sign(coefB)·mB` 사분면 feasibility, 충돌시 1D 폴백. 검증: 위반 step 0. |
| ◐ | **Multi-class** | 중 | ✅ OvR(클래스별 이진 부스터, 행 정규화) 완료 — iris acc 1.0. ☐ 네이티브 softmax(K-출력)는 후속. |
| ☐ | **SHAP 유사 해석** | 중~높 | TreeSHAP는 axis-tree 가정 → oblique엔 직접 적용 불가. 경로기여 근사 또는 KernelSHAP 래핑. |
| ☐ | **Native categorical** | 중~높 | ablation상 continuous가 이김 → 고카디널리티 전용 set-partition(LightGBM식 gradient 정렬) 검토. 신중히. |
| ☑ | **Incremental training** | 높 | ✅ 완료. `warm_start=True` + `n_estimators`↑ → 기존 트리 위에 추가분만 학습(`fit_more`). raw 상태는 저장 트리로 재구성(상태 저장 불필요), rng 멤버화로 시퀀스 연속. 이진/회귀/다중클래스(OvR) 지원. subsample=1서 scratch와 bit-identical 검증. |
| ☐ | **Out-of-core** | 높 | 데이터 청크 스트리밍 + 디스크 binning. 대용량용. |
| ☐ | **GPU** | 매우 높 | 쌍탐색/히스토그램 CUDA 커널. 별도 백엔드. 마지막. |

## 권장 진행 순서
1. **serialization + feature importance + regression 안정화** (낮은 난이도, 즉시 가치).
2. **multi-class + monotonic** (표 데이터 실사용 필수 기능).
3. **SHAP 해석 + native categorical** (차별화).
4. **incremental / out-of-core / GPU** (스케일).
