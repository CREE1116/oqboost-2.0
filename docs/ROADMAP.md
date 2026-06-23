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
| ☐ | **Regression 안정화** | 낮음 | 현 squared-error에 Huber/quantile 손실, 출력 클리핑, init=median 옵션. |
| ☐ | **Thread scaling** | 중 | 현 노드별 쌍병렬은 small-data서 fork-join 한계. 트리내부/배치 predict 병렬 개선, OMP 스케줄 튜닝. |
| ☐ | **Monotonic constraints** | 중 | 피처별 단조 제약 → oblique서 까다로움(선형결합 부호 제약). 1D 분할엔 쉬움, 2D는 coef 부호 제약. |
| ☐ | **Multi-class** | 중 | one-vs-rest 또는 softmax(K 부스터). raw score K개 + 멀티 gradient. API/wrapper 확장. |
| ☐ | **SHAP 유사 해석** | 중~높 | TreeSHAP는 axis-tree 가정 → oblique엔 직접 적용 불가. 경로기여 근사 또는 KernelSHAP 래핑. |
| ☐ | **Native categorical** | 중~높 | ablation상 continuous가 이김 → 고카디널리티 전용 set-partition(LightGBM식 gradient 정렬) 검토. 신중히. |
| ☐ | **Incremental training** | 높 | warm-start로 트리 추가(`n_estimators` 증가분만 학습). 상태 보존 fit. |
| ☐ | **Out-of-core** | 높 | 데이터 청크 스트리밍 + 디스크 binning. 대용량용. |
| ☐ | **GPU** | 매우 높 | 쌍탐색/히스토그램 CUDA 커널. 별도 백엔드. 마지막. |

## 권장 진행 순서
1. **serialization + feature importance + regression 안정화** (낮은 난이도, 즉시 가치).
2. **multi-class + monotonic** (표 데이터 실사용 필수 기능).
3. **SHAP 해석 + native categorical** (차별화).
4. **incremental / out-of-core / GPU** (스케일).
