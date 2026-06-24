# OQBoost 2.0 — Feature Roadmap

결과론적 목표(차차 업데이트). 난이도·의존성 고려한 제안 순서.

## 현재 (구현됨)

- 2D-oblique GBDT, 히스토그램 binning, OpenMP 쌍병렬, fast_dir(gradient 회귀 방향, 기본).
- 이진 분류 + 회귀, scikit-learn API, **native NaN(결측치)**.
- 다중플랫폼 wheel CI(cibuildwheel), Optuna 벤치(AUC/acc/bacc/train·infer time).

## 로드맵

| #   | 기능                                    | 난이도  | 메모                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --- | --------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------------------ | --- | --- | --- | -------------------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------------------- |
| ☑   | **Model serialization**                 | 낮음    | ✅ 완료. C++ `serialize/deserialize`(Node POD memcpy) + 래퍼 `__getstate__/__setstate__` → pickle·joblib 호환.                                                                                                                                                                                                                                                                                                                                |
| ☑   | **Feature importance**                  | 낮음    | ✅ 완료. 채택 분할 gain 피처별 누적 → `feature_importances_`(정규화), pickle 보존.                                                                                                                                                                                                                                                                                                                                                            |
| ☑   | **분류 threshold 튜닝**                 | 낮음    | ✅ 완료. `threshold="balanced"/"f1"` → holdout서 cut 최적화(`decision_threshold_`). proba는 calibrated(mean≈base rate)이나 불균형서 0.5 cut은 bacc 붕괴 → 옮겨야 함. 기본 0.5 유지.                                                                                                                                                                                                                                                           |
| ☑   | **Regression 안정화**                   | 낮음    | ✅ 완료. `loss="huber"/"quantile"`(robust 손실), `clip`(예측 train 범위 clamp), huber/quantile은 init=median 자동. quantile은 leaf-value line-search(분위수 재계산). 이상치 train서 huber/median MAE ≪ squared. 한계: 극단 분위(α≈0.9) interval은 shallow tree서 안쪽 편향(median은 정확).                                                                                                                                                    |
| ☑   | **Thread scaling**                      | 중      | ✅ 완료. 작업량(쌍수×표본수) 임계 이하 노드는 serial 폴백(`if` clause)으로 fork-join 오버헤드 제거 — small-data 8스레드 회귀(0.105→0.081s) 해소. gradient/raw/predict n-루프도 임계 가드. big-data 4코어 ~2.8x(70%) 유지, 스레드 수 무관 bit-identical.                                                                                                                                                                                       |
| ☑   | **Monotonic constraints**               | 중      | ✅ 완료. `monotone_constraints`(-1/0/+1, list 또는 dict). 풀 oblique 단조: 고정 타피처 직선 위 사선분할=단일 threshold라 axis 기법(중점 값-경계 전파 + leaf clamp)이 이식됨. 쌍 둘다 제약시 `sign(coefA)·mA==sign(coefB)·mB` 사분면 feasibility, 충돌시 1D 폴백. 검증: 위반 step 0.                                                                                                                                                           |
| ◐   | **Multi-class**                         | 중      | ✅ OvR(클래스별 이진 부스터, 행 정규화) 완료 — iris acc 1.0. ☐ 네이티브 softmax(K-출력)는 후속.                                                                                                                                                                                                                                                                                                                                               |
| ☑   | **OQBoost 네이티브 설명**               | 중~높   | ✅ 완료. TreeSHAP 복제 대신 oblique 고유 설명: `coefficient_importances_`(Σ gain·                                                                                                                                                                                                                                                                                                                                                             | coef | ), `interaction_importances_`(d×d, Σ gain· | a   | ·   | b   | — 사선쌍 상호작용, 비용 0), `explain(x)`(트리기여 lr·w를 경로 피처에 gain· | coef | 비율 분배 → **Σφ=예측−base 가산적**, SHAP 직접 비교 가능). 이진·회귀·**다중클래스(OvR, (n,K,d))** 지원. |
| ☑   | **Native categorical**                  | 중~높   | ✅ 완료. `categorical_features`(인덱스/bool 마스크) → 그 피처만 **무손실 비닝**(레벨당 1 bin, max_bins 무관). 근본원인: 정수코드를 연속처럼 quantile 비닝 → card>max_bins서 distinct 레벨 병합 손실. 무손실 비닝이 연속 저해상도(방향 안정) 유지하며 고card 명목 보존. 합성 고card-범주신호서 +0.16 AUC(card200: 0.65→0.81). (adult는 명목이 저card라 효과 미미 — 데이터별.)                                                                  |
| ☑   | **Incremental training**                | 높      | ✅ 완료. `warm_start=True` + `n_estimators`↑ → 기존 트리 위에 추가분만 학습(`fit_more`). raw 상태는 저장 트리로 재구성(상태 저장 불필요), rng 멤버화로 시퀀스 연속. 이진/회귀/다중클래스(OvR) 지원. subsample=1서 scratch와 bit-identical 검증.                                                                                                                                                                                               |
| ☐   | **Out-of-core**                         | 높      | 데이터 청크 스트리밍 + 디스크 binning. 대용량용.                                                                                                                                                                                                                                                                                                                                                                                              |
| ☐   | **GPU**                                 | 매우 높 | 쌍탐색/히스토그램 CUDA 커널. 별도 백엔드. 마지막.                                                                                                                                                                                                                                                                                                                                                                                             |
| ◐   | **LOB (Lineage Oblique Boosting, 3.0)** | 높      | 실험적 opt-in `max_lineage>0`. **2×2 solve만으로 고차원 oblique 상호작용을 근사** — 노드가 조상 방향 z 상속 → `(z,raw)`,`(z,z)` 쌍 탐색 → 방향 계층 합성(d-차원 직접 최적화 없이 2D로 쌓음). dense 방향 dirs\_ 테이블 + dir_id 노드, classic(0) 무영향. root-전수 + 깊은노드 SIS가 검증된 스크리닝. 고차원 oblique+상호작용 구조서 이득(합성XOR +0.02). axis-tree 불가능 = oblique 전용. ☐ 후속: lineage-aware 스크리닝 고도화, explain 지원. |
| ☑   | **1D 경쟁 제거**                        | 낮음    | ✅ 완료. 측정상 1D vs 2D 경쟁 제거해도 mean Δ −0.0003(phoneme 21% 1D조차 무손실) — 2D가 b≈0으로 1D 표현. eval_1d는 2D가 분할 못 찾는 퇴화 노드 폴백으로만. 코어 단순화 + 노드당 s1 생략.                                                                                                                                                                                                                                                      |

## 라이브러리 성숙도 (다음 우선순위)

알고리즘 코어는 성숙(정확도 = GBDT 패러다임 천장, exotic 트릭 LOB/linear-leaf/ordered 다 ~0.01서 막힘).
남은 건 **소프트웨어 완성도** — <1주 모델이라 표준 기능·sklearn 준수에 구멍. (2026-06-25 검증)

| #   | 항목                          | 난이도 | 메모                                                                                                          |
| --- | ----------------------------- | ------ | ------------------------------------------------------------------------------------------------------------- |
| ☑   | **sample_weight**             | 중     | ✅ 완료. `fit(X, y, sample_weight)` → boost_rounds서 g,h에 w 곱(Newton 정확) + 가중 init. 검증: w=1 항등, 극단가중 지배, weighted≈replicated(mean diff ~0.002; 잔차=unweighted binning 2차효과). 다중클래스/회귀/warm-start. |
| ☐   | **early stopping / eval_set** | 중     | `eval_set` + `n_iter_no_change` → val 메트릭 모니터링, `best_iteration_`서 트리 절단. n_estimators 수동 탈피. |
| ☐   | **sklearn 완전 준수**         | 낮~중  | `check_estimator` FAIL — predict서 n*features 일관성 검증 X, `feature_names_in*` 없음. 신뢰성·생태계 호환.    |
| ☐   | **class_weight**              | 낮     | 분류 클래스 가중(sample_weight 위에 구축).                                                                    |
| ☐   | **sparse 입력**               | 중     | scipy sparse `X` 지원 (현재 dense만). 고차원 희소 데이터.                                                     |
| ☐   | **테스트 스위트 / CI 품질**   | 중     | pytest 커버리지(엣지: d=1, 단일클래스, 극단값, 직렬화 버전호환), property test. 회귀 방지.                    |
| ☐   | **기본값 튜닝 / 문서**        | 낮     | untuned 성능 좋게 defaults 점검, 사용 예제·노트북.                                                            |

## 스케일 (장기)

| #   | 항목        | 난이도  | 메모                                                     |
| --- | ----------- | ------- | -------------------------------------------------------- |
| ☐   | Out-of-core | 높      | 데이터 청크 스트리밍 + 디스크 binning. 대용량용.         |
| ☐   | GPU         | 매우 높 | 쌍탐색/히스토그램 CUDA 커널. 별도 백엔드. 마지막.        |
| ◐   | LOB 고도화  | 높      | lineage-aware 스크리닝, explain 지원 (실효 낮아 후순위). |

## 권장 진행 순서

1. **sample_weight → class_weight** (가장 많이 쓰이는 누락 기능).
2. **early stopping + eval_set** (실사용 핵심 UX).
3. **sklearn 완전 준수 + 테스트 스위트** (신뢰성·생태계).
4. **sparse 입력 / 기본값·문서**.
5. **스케일(out-of-core / GPU)** — 수요 생기면.
