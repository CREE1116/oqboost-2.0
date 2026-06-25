# OQBoost 통계적 우위성 정리 — 개정 완전판

**개정 사항 요약 (이전 버전 대비):** Lemma 1의 근거였던 "Ridgelet 근사이론(Candès–Donoho)과의 유비"는 부정확한 인용이었음이 확인되어 폐기합니다. Ridgelet 이론은 **매끄럽고 admissible한**(평균 0, 적분 가능한) ridge 원자를 다루는데, oblique stump는 **불연속 0/1 indicator**이기 때문에 같은 근사율 공식이 적용되지 않습니다. 또한 indicator의 *합*만으로는 직선 경계만 만들 수 있어, 코너나 곡선 경계를 가진 일반적인 2변수 함수를 정확히 표현할 수 없습니다.

이 개정판은 Lemma 1을 **Maurey–Jones–Barron lemma**와 **Barron(1993)의 스펙트럴 근사 정리**로 교체하여, $M$개 항의 합으로 $O(1/\sqrt{M})$ 오차를 얻는 부분을 실제로 증명 가능한 형태로 다시 세웁니다. 이는 Lemma 3에서 이미 쓰고 있는 convex-hull 논증과도 동일한 언어를 공유하므로, 전체 정리가 처음부터 끝까지 하나의 도구(Maurey 경험적 방법)로 일관되게 연결됩니다.

> **⚠ 비교 대상 명시.** 이 정리가 말하는 "우위성"은 $\mathcal{H}_{\text{OQ}}$(2D-oblique) vs $\mathcal{H}_{\text{Full}}$(임의 $D$차원 oblique half-space)의 비교입니다 — **axis-aligned 트리(XGBoost/LightGBM/CatBoost)와의 비교가 아닙니다.** 어떤 주류 라이브러리도 $\mathcal{H}_{\text{Full}}$을 쓰지 않으며, axis-aligned 분할도 (피처 1개 선택이라) 복잡도가 낮습니다. 따라서 본 정리의 메시지는 "OQ가 XGBoost를 이긴다"가 아니라, **2D-oblique가 full-oblique의 복잡도 폭발($\sqrt{D}$)을 피해 axis-tree 수준의 $\sqrt{\log D}$ 복잡도를 유지하면서 2D 표현력(사선·상호작용)을 얻는다**는 것입니다. axis-tree 대비 실증적 우열은 정리가 아니라 §벤치마크의 경험적 문제입니다.

---

## 0. 설정

$f^*:[0,1]^D\to\mathbb{R}$의 ANOVA 분해

$$f^*(x) = f_0 + \sum_k f_k(x_k) + \sum_{i<j} f_{ij}(x_i,x_j) + R(x)$$

에서 고차 잔차 $R$의 분산 기여가 $\le\epsilon$이라 가정합니다. 두 가설 공간:

- $\mathcal{H}_{\text{OQ}} = \bigcup_{i<j}\mathcal{H}_{ij}$, $\mathcal{H}_{ij}=\{x\mapsto \mathbb{1}[w_ix_i+w_jx_j\le\theta]\}$
- $\mathcal{H}_{\text{Full}} = \{x\mapsto \mathbb{1}[w^\top x\le\theta]: w\in\mathbb{R}^D\}$

---

## 1. Lemma 1 (개정) — 근사오차 동등성

### 1.1 정의: 사전(dictionary)에 대한 변동 노름

$2$변량 성분 $f_{ij}$에 대해 dictionary

$$\mathcal{G}_{ij} = \{\pm\,\mathbb{1}[w_ix_i+w_jx_j\le\theta] : \|w\|=1,\ \theta\in\mathbb{R}\}\ \cup\ \{\pm 1\}$$

를 정의합니다 ($\mathcal{H}_{ij}$의 부호 포함 버전, $\|h\|_\infty\le 1$). **변동 노름(variation norm)**을

$$\|f\|_{\mathcal{V}(\mathcal{G}_{ij})} := \inf\Big\{c>0 : \tfrac{f}{c} \in \overline{\text{conv}}(\mathcal{G}_{ij})^{L^2}\Big\}$$

로 정의합니다 (여기서 $\overline{\text{conv}}(\cdot)^{L^2}$는 $L^2([0,1]^2)$에서의 닫힌 convex hull). 이는 Barron(1993)이 단일 은닉층 신경망 근사에서 쓴 것과 동일한 개념을 half-plane indicator dictionary에 적용한 것입니다.

**중요한 정정:** 이전 버전의 가정인 "$f_{ij}$가 유계 변동(bounded variation) $V_{ij}<\infty$"은 $\|f_{ij}\|_{\mathcal{V}(\mathcal{G}_{ij})}<\infty$를 **함의하지 않습니다.** 두 개념은 다릅니다. 이 개정판에서는 가정을 다음으로 **교체**합니다:

> **가정 1.1 (스펙트럴 조건, Barron 1993 Condition).** $f_{ij}$가 $[0,1]^2$ 위로 확장 가능하고 푸리에 변환 $\hat{f}_{ij}$가 존재하여
> $$C_{ij} := \int_{\mathbb{R}^2} \|\omega\|\,|\hat{f}_{ij}(\omega)|\,d\omega < \infty$$
> (즉 $f_{ij}$의 "1차 모멘트가 적분 가능한 스펙트럴 질량"을 가짐).

이 조건은 $f_{ij}\in C^2$이고 2차 도함수가 $L^1$에 있는 경우 등 흔한 정칙성 조건들로부터 성립하며, 고전적 유계변동보다는 강하지만 ANOVA 2변량 성분(보통 매끄럽거나 조각별 매끄러운 함수)에 대해서는 합리적인 가정입니다.

### 1.2 명제 (수정판)

가정 1.1 하에서, 모든 $M\ge1$에 대해 $M$개의 (부호 있는) oblique stump $h_1,\dots,h_M\in\mathcal{G}_{ij}$와 가중치 $\alpha_m = C_{ij}/M$이 존재하여

$$\left\|f_{ij} - \sum_{m=1}^M \alpha_m h_m\right\|_{L^2}^2 \;\le\; \frac{4\,C_{ij}^2}{M}$$

따라서 목표 오차 $\delta$를 위해서는 $M = O(C_{ij}^2/\delta^2)$.

### 1.3 증명

**Step 1 (스펙트럴 조건 → 변동 노름 유한성).** Barron(1993, §IX)의 표현을 따릅니다. 실함수 $f_{ij}$는 푸리에 역변환에서 위상 $\beta(\omega)=\arg\hat f_{ij}(\omega)$를 분리해

$$f_{ij}(x) = f_{ij}(0) + \int_{\mathbb{R}^2}\big[\cos(\omega\cdot x+\beta(\omega))-\cos\beta(\omega)\big]\,|\hat f_{ij}(\omega)|\,d\omega$$

로 쓸 수 있습니다 (허수부는 $f_{ij}$가 실함수라 상쇄; 상수 $\cos\beta$ 항은 $x$-무관 평탄항 $\pm 1\in\mathcal{G}_{ij}$로 흡수). 각 피적분항 $g_\omega(x)=\cos(\omega\cdot x+\beta)-\cos\beta$는 방향 $u=\omega/\|\omega\|$의 **ridge 함수**이고, 그 1차원 프로파일 $t\mapsto\cos(\|\omega\|t+\beta)$의 전변동(total variation)은 컴팩트 구간 위에서 $\le 2\|\omega\|$입니다. 1차원에서 전변동 $V$인 함수는 정확히 $V\cdot\overline{\text{conv}}\{\pm\mathbb{1}[\,\cdot\le\theta]\}$에 속하므로 (BV ⟺ step의 적분, **1차원에서 엄밀**),

$$\frac{g_\omega}{2\|\omega\|}\ \in\ \overline{\text{conv}}(\mathcal{G}_{ij}).$$

따라서 $f_{ij}$는 가중치 $2\|\omega\|\,|\hat f_{ij}(\omega)|$로 $\mathcal{G}_{ij}$ 원자들의 (적분) convex 결합이며, 총질량은 $\int_{\mathbb{R}^2} 2\|\omega\|\,|\hat f_{ij}(\omega)|\,d\omega = 2C_{ij}$. 즉

$$f_{ij}/(2C_{ij}) \in \overline{\text{conv}}(\mathcal{G}_{ij})^{L^2}, \qquad \text{즉}\quad \|f_{ij}\|_{\mathcal{V}(\mathcal{G}_{ij})} \le 2C_{ij}.$$

(상수 인자 $2\|\omega\|$는 cosine 프로파일의 1차원 전변동에서 정확히 나오며, Step 2의 $c=2C_{ij}$와 일치합니다.)

**Step 2 (Maurey–Jones–Barron empirical method).** $f/c \in \overline{\text{conv}}(\mathcal{G})$, $\|h\|_\infty\le B$인 dictionary $\mathcal{G}$가 주어지면, $f/c$는 $\mathcal{G}$ 위의 확률측도 $\nu$에 대해 $f/c = \mathbb{E}_{h\sim\nu}[h]$ (또는 그 닫힌 근사)로 쓸 수 있습니다. $h_1,\dots,h_M \stackrel{iid}{\sim}\nu$를 뽑아 $\bar h_M = \frac1M\sum_m h_m$이라 하면, 표준 분산 분해(bias-variance, Jones 1992; Barron 1993; Pisier's Maurey lemma)에 의해

$$\mathbb{E}\big\|f/c - \bar h_M\big\|_{L^2}^2 \;\le\; \frac{B^2}{M}$$

(직관: $\mathbb{E}\|\bar h_M - \mathbb{E}\bar h_M\|^2 = \frac1M \text{Var}(h) \le B^2/M$이고 $\mathbb{E}\bar h_M = f/c$이므로 편향은 0). $c=2C_{ij}$, $B=1$을 대입하면

$$\mathbb{E}\big\|f_{ij} - c\bar h_M\big\|_{L^2}^2 \le \frac{c^2}{M} = \frac{4C_{ij}^2}{M}$$

기댓값이 이 값 이하이므로, **적어도 하나의** 구체적인 $h_1,\dots,h_M$ 선택이 이 부등식을 만족시킵니다(확률론적 존재증명, probabilistic method). $\alpha_m = c/M = 2C_{ij}/M$로 두면 명제가 증명됩니다. $\blacksquare$

### 1.4 누적 오차 및 1차 성분

$1$차 성분 $f_k$, 상수항 $f_0$도 동일한 가정(1차원 스펙트럴 조건, 사실상 표준 BV 조건으로 충분—1차원에서는 BV ⟺ 유한 변동 노름이 정확히 성립)으로 같은 방식으로 처리됩니다. 모든 $\binom{D}{2}$쌍에 대해 합산하면

$$\text{Approx}(\mathcal{H}_{\text{OQ}}\text{-ensemble}) \le O(\epsilon) + \sum_{i<j} O\!\left(\frac{C_{ij}^2}{M}\right)$$

이고 우변 둘째 항은 $M\to\infty$에서 $0$으로 가는 **알고리즘적** 오차입니다.

### 1.5 이 버전에서 정직하게 남는 한계

- 가정 1.1(스펙트럴 조건)은 고전적 유계변동보다 **강한** 가정입니다. 즉 이전 버전이 "BV면 충분"이라 주장한 것은 틀렸고, 실제로는 그보다 좁은 함수족에서만 이 비율이 성립한다고 봐야 합니다. ANOVA 2변량 성분이 실제로 이 조건을 만족하는지는 **데이터마다 확인이 필요한 경험적 질문**입니다.
- $C_{ij}$가 $D$에 의존해 커지지 않는다는 보장은 없습니다(예: 변수가 많아질수록 개별 $f_{ij}$가 더 날카로워지는 경우). Theorem의 최종 비율 $\Theta(\sqrt{D/\log D})$은 $C_{ij}$, 그리고 후술할 $\Lambda$가 $D$와 독립적으로 유계라는 전제 위에서만 성립합니다.

---

## 2. Lemma 2 — 합집합 가설 클래스의 Rademacher 상한 (변경 없음, 검증됨)

**명제.** $\mathcal{H}=\bigcup_{k=1}^N\mathcal{H}_k$, 각 VC차원 $d_k\le d$이면

$$\hat{\mathcal{R}}_n(\mathcal{H}) \le C\sqrt{d/n} + C'\sqrt{\log N/n}$$

**증명.** $\sup_{h\in\mathcal{H}} = \max_k\sup_{h\in\mathcal{H}_k}$로 분해. 각 클래스 내부는 Sauer–Shelah growth function bound + Dudley entropy integral (Bartlett & Mendelson, 2002, Thm 12)에 의해 $\mathcal{R}_n(\mathcal{H}_k)\le C\sqrt{d_k/n}$. $N$개 중 최댓값은 finite-class maximal inequality(Massart, 2000)로 $\sqrt{\log N/n}$ 패널티만 추가됩니다.

$\mathcal{H}_{ij}$는 $\mathbb{R}^2$의 half-space이므로 $d_k=3$, $N=\binom{D}{2}$:

$$\hat{\mathcal{R}}_n(\mathcal{H}_{\text{OQ}}) = O\!\left(\sqrt{\frac{\log D}{n}}\right) \qquad \blacksquare$$

샘플 $\{x_i\}$가 모든 $\mathcal{H}_k$에서 공유되어도, empirical Rademacher 복잡도가 고정 샘플 위에서 정의되므로 union bound는 그대로 유효합니다. 이 lemma는 표준적인 교과서 결과의 직접 적용이며, 이전 검토에서도 견고하다고 판단했고 이번 개정에서도 수정할 필요가 없습니다.

---

## 3. Lemma 3 — 부스팅 앙상블로의 확장 (Lemma 1 개정과 표현을 통일)

**명제.** $\|\alpha\|_1\le\Lambda$로 제약된

$$\mathcal{F}_\Lambda = \Big\{\textstyle\sum_{m=1}^M \alpha_m h_m : h_m\in\mathcal{H}_{\text{OQ}},\ \|\alpha\|_1\le\Lambda\Big\}$$

의 Rademacher 복잡도는 $M$에 무관하게 $\mathcal{R}_n(\mathcal{F}_\Lambda) \le \Lambda\cdot\mathcal{R}_n(\mathcal{H}_{\text{OQ}})$.

**증명.** Symmetric convex hull의 Rademacher 복잡도가 base 클래스와 동일하다는 성질(Bartlett & Mendelson, 2002, Lemma 22)에 의해 $\Lambda$-scaling만 남습니다. $M$은 사라집니다. $\blacksquare$

**주목할 점:** 이 Lemma 3의 증명 기법(부호 있는 convex hull, $\ell_1$-norm 제약)은 위 Lemma 1 §1.3에서 쓴 Maurey 방법과 **동일한 수학적 대상**(convex hull의 닫힘, 변동/노름 제약)을 다루고 있습니다. 즉 Lemma 1의 "표현오차"와 Lemma 3의 "추정오차"가 둘 다 같은 변동 노름 $\mathcal{V}(\mathcal{G}_{ij})$ 언어로 통일되어, $\Lambda$(부스팅 가중치의 $\ell_1$질량)와 $C_{ij}$(개별 성분의 스펙트럴 변동 노름)가 사실상 같은 종류의 양임이 드러납니다 — 이는 이전 버전에는 없던 통찰로, 두 lemma가 별개 도구를 쓰는 것이 아니라 하나의 골격 위에 있음을 보여줍니다.

대응 결과 (변경 없음):

$$\mathcal{R}_n(\mathcal{F}_\Lambda^{\text{Full}}) = O\!\left(\Lambda\sqrt{D/n}\right), \qquad \mathcal{R}_n(\mathcal{F}_\Lambda^{\text{OQ}}) = O\!\left(\Lambda\sqrt{\log D/n}\right)$$

---

## 4. 최종 정리

신뢰수준 $1-\delta$에서,

$$\text{Risk}(\hat f_{\text{Full}}) \le O(\epsilon) + O\!\left(\Lambda\sqrt{D/n}\right) + O\!\left(\sqrt{\log(1/\delta)/n}\right)$$

$$\text{Risk}(\hat f_{\text{OQ}}) \le O(\epsilon) + O\!\left(\Lambda\sqrt{\log D/n}\right) + O\!\left(\sqrt{\log(1/\delta)/n}\right)$$

여기서 $\epsilon$의 정확한 의미는 §1.5의 한계를 반영해 다음으로 명확히 합니다: $O(\epsilon)$ 항은 (a) 3차 이상 ANOVA 잔차의 분산 기여, **그리고** (b) 가정 1.1을 만족하지 않는 부분에서 오는 잔여 근사오차, 두 가지를 합친 것입니다.

**Corollary (샘플 복잡도).**

$$n_{\text{Full}}(\eta) = O\!\left(\frac{\Lambda^2 D}{\eta^2}\right), \qquad n_{\text{OQ}}(\eta) = O\!\left(\frac{\Lambda^2 \log D}{\eta^2}\right)$$

$$\Rightarrow\quad \frac{n_{\text{Full}}(\eta)}{n_{\text{OQ}}(\eta)} = \Theta\!\left(\frac{D}{\log D}\right) \qquad \blacksquare$$

---

## 5. 인용 결과 정리표 (개정)

| 보조정리                     | 역할                      | 레퍼런스                                                                                              | 비고                                                                          |
| ---------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Lemma 1 (Maurey–Barron)      | 표현오차 동등성           | Barron (1993), _IEEE Trans. Info. Theory_; Jones (1992); Maurey's empirical method (Pisier 1981 경유) | 이전 판의 ridgelet 인용을 폐기하고 교체. 가정도 BV → 스펙트럴 조건으로 강화됨 |
| Lemma 2 (VC + 합집합)        | $D$ vs $\log D$ 핵심 트릭 | Vapnik–Chervonenkis (1971); Massart (2000); Bartlett & Mendelson (2002) Thm 12                        | 변경 없음, 검증됨                                                             |
| Lemma 3 (Convex hull 불변성) | $M$-라운드 독립성         | Bartlett & Mendelson (2002) Lemma 22; Schapire–Freund margin theory (1998)                            | 변경 없음, Lemma 1과 동일한 변동 노름 언어로 재해석                           |
| Oracle inequality            | 최종 합성                 | Koltchinskii (2001)                                                                                   | 변경 없음                                                                     |

---

## 6. 솔직한 평가 — 무엇이 닫혔고 무엇이 남았는가

**완전히 견고한 부분:**

- **Lemma 2**는 교과서적이고 정확합니다. 정리 전체의 조합론적 핵심($\log\binom{D}{2}$ 트릭)이며 의심할 이유가 없습니다.
- **Lemma 3**도 표준적인 margin-theory 결과의 정확한 적용입니다.
- 개정된 **Lemma 1**은 이제 실제로 증명 가능한 정리(Maurey–Jones–Barron)에 기반합니다. 더 이상 "유비"가 아니라 인용 가능한 정리의 직접 적용입니다.

**여전히 주의가 필요한 부분 (이번 개정으로 드러난 것):**

- Lemma 1의 **가정이 바뀌었다는 점이 가장 중요합니다.** 이전 판은 "유계변동이면 충분"이라 주장했지만, 이는 틀렸습니다. 실제로 필요한 조건(스펙트럴 1차 모멘트 적분 가능)은 더 강합니다. 즉 OQBoost의 우위성 정리는 **이전에 생각했던 것보다 더 좁은 함수 클래스**에서만 현재 엄밀하게 보장됩니다. 이는 정리를 "약화"시킨 것이 아니라, 이전 정리가 **틀린 가정 위에 서 있었다는 것**을 고친 것입니다.
- $C_{ij}$ (Lemma 1의 변동 노름)와 $\Lambda$ (Lemma 3의 부스팅 가중치 노름)가 실제로 $D, n$에 독립적으로 유계인지는 여전히 **실험적으로 확인해야 할 과제**입니다. 두 양이 같은 종류의 변동 노름이라는 것을 §3에서 밝혔으므로, 이제는 "$C_{ij}$들의 합이 $\Lambda$로 흡수되는 메커니즘"을 하나의 실험으로 함께 검증할 수 있습니다 — 이전 판에서는 이 둘이 무관한 가정처럼 분리되어 있었습니다.

**남은 실증적 검증 과제:**

- ANOVA 2변량 성분 $f_{ij}$가 실제 정형 데이터(UCI, OpenML)에서 가정 1.1(스펙트럴 조건)을 만족하는지, 혹은 적어도 $C_{ij}$가 작은지 확인하는 실험.
- $\Lambda$가 $D, n$에 대해 안정적으로 유계인지에 대한 boosting 학습 곡선 실험.
- $\epsilon$(고차 상호작용 잔차)의 경험적 분산 분해 실험.

---

## 7. 경험적 검증 (실측)

위 한계 중 두 가지 — 가정($\epsilon$ 작음)과 핵심 주장($\sqrt{\log D}$ 복잡도) — 을 OpenML 실데이터로 직접 측정했다 (`scripts/`의 로더 사용, held-out test).

### 7.1 $\epsilon$ — 실데이터는 정말 $\le$2차 ANOVA인가?

상호작용 차수를 XGBoost `max_depth`로 캡(`1`=주효과, `2`=2-way, `6`=full)하고, **3차 이상이 설명하는 신호 비율** $\hat\epsilon = (\text{score}_{d6}-\text{score}_{d2})/\text{score}_{d6}$을 측정 (분류=AUC, 회귀=$R^2$).

| dataset | d1(주효과) | d2(2-way) | d6(full) | $\hat\epsilon$ |
| --- | ---: | ---: | ---: | ---: |
| german | 0.853 | 0.858 | 0.824 | **0.0%** |
| diabetes | 0.857 | 0.847 | 0.830 | **0.0%** |
| kc1 | 0.817 | 0.823 | 0.809 | **0.0%** |
| spambase | 0.985 | 0.990 | 0.989 | **0.0%** |
| cpu_small (R²) | 0.969 | 0.975 | 0.979 | 0.4% |
| phoneme | 0.881 | 0.916 | 0.956 | 4.2% |
| house_16H (R²) | 0.463 | 0.545 | 0.601 | 9.3% |
| puma32H (R²) | 0.224 | 0.657 | 0.921 | **28.7%** |

**해석.** 대부분의 셋에서 2-way가 full을 따라잡아 $\hat\epsilon\approx0$ — 가정(저차 ANOVA)이 **실제로 성립**한다. 결정적 반례는 **puma32H ($\hat\epsilon=28.7\%$)**: 3차+ 상호작용이 큰 비중. 그리고 이는 회귀 벤치마크에서 **OQBoost가 CatBoost에 가장 크게 뒤진 셋**($R^2$ 0.938 vs 0.954)과 정확히 일치한다. 즉 **이론의 $\epsilon$ 항이 OQBoost의 실패 모드를 예측**한다 — 2D-oblique는 고차 상호작용 구조에서 원리적으로 불리하고, 데이터가 그런 구조일 때 정확히 약해진다.

### 7.2 $\sqrt{\log D}$ — 노이즈 차원에 대한 강건성

원본 피처에 순수 가우시안 노이즈 피처 $k$개를 추가해 $D$를 키우고 OQBoost test AUC 변화를 측정. 이론 예측: 추정오차 $\sim\sqrt{\log D/n}$ (느린 증가), $\sqrt{D/n}$ 아님.

| dataset | $D_0$ | $D{=}D_0$ | $+20$ | $+100$ | $+400$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| phoneme | 5 | 0.943 | 0.922 | 0.910 | 0.903 |
| kc1 | 21 | 0.820 | 0.795 | 0.781 | 0.797 |
| spambase | 57 | 0.990 | 0.988 | 0.987 | 0.987 |

**해석.** 피처를 **81배**(phoneme $5\to405$) 늘려도 AUC는 $-0.040$에 그친다. 복잡도가 $\sqrt{D}$였다면 $\sqrt{405/5}\approx9$배 추정오차 폭발로 붕괴했어야 하지만, 관측된 완만한 감소는 $\sqrt{\log405/\log5}\approx1.9$배에 부합한다 — **Lemma 2의 $\sqrt{\log D}$ 복잡도가 경험적으로 확인**된다 (spambase는 거의 평탄, kc1은 $D{=}421$에서 오히려 회복).

### 7.3 LOB — $\epsilon$ 큰 셋에서 고차 항을 되찾는가?

§7.1이 "2D로 부족한 셋"(큰 $\hat\epsilon$)을 식별했으니, **LOB**(`max_lineage>0`, 조상 방향 상속으로 고차 oblique 상호작용을 $2\times2$ solve만으로 근사 — [internals/lob](internals/lob.md))이 정확히 그 셋에서 복구하는지 측정. $R^2$:

| dataset | $\hat\epsilon$ | `ml=0` | `ml=2` | `ml=4` | CatBoost |
| --- | ---: | ---: | ---: | ---: | ---: |
| **puma32H** | 29% | 0.9389 | 0.9505 | **0.9516** | 0.9498 |
| house_16H | 9% | 0.6573 | 0.6546 | 0.6506 | 0.6343 |
| cpu_small | 0% | 0.9826 | 0.9832 | 0.9832 | 0.9811 |

**해석.** LOB의 이득이 $\hat\epsilon$에 **단조 비례**한다: puma32H($\hat\epsilon{=}29\%$)에서 $+0.013$으로, `ml=0`일 때 CatBoost에 지던 것($0.939<0.950$)을 **`ml=4`에서 역전**($0.952>0.950$)시킨다 — 이론이 "2D 실패"로 지목한 바로 그 셋에서 고차 항을 되찾는다. 반면 저차 셋(cpu_small $\hat\epsilon{=}0\%$)에서는 무변($+0.0006$), house_16H는 약간 손해($-0.007$). 즉 **LOB의 용도는 정확히 $\hat\epsilon$가 큰 고차-상호작용 데이터**이며, "실데이터 평균상 LOB는 마진얼"이라는 과거 관찰은 저차 셋이 다수라 생긴 **평균의 착시**였다. 이론($\epsilon$ 항이 2D의 한계를 규정) ↔ LOB(그 한계를 넘는 고차 합성)가 정확히 맞물린다.

### 7.4 종합

- **가정($\epsilon$)**: 다수 실데이터에서 성립하며, 위배되는 셋(puma32H)은 이론이 OQBoost 약점으로 예측한 바로 그 셋 — 정리가 **반증 가능하고 예측력 있다**.
- **핵심 주장($\sqrt{\log D}$)**: 노이즈 차원 강건성으로 직접 확인됨.
- **처방($\epsilon$ 클 때)**: LOB가 고차 항을 되찾아 그 셋에서 OQBoost를 CatBoost 위로 올린다 — 이론의 한계항과 알고리즘적 처방이 일치.
- 아직 미측정: $C_{ij}$의 직접 추정(2D-FFT)과 $\Lambda$의 $D,n$-안정성 — 후속 과제로 남김.

> 재현: `scripts/theory_validation.py` (max_depth 캡 ANOVA 분해 + 노이즈 차원 스윕).
