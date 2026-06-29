// oqboost_core.cpp — OQBoost core (histogram-binned 2D-oblique GBDT)
// 2D-oblique Newton-boosted GBDT. 전역 사전 binning(히스토그램 트릭)으로 노드별
// 정렬 제거. 범주는 기본 연속(정수코드) + 선택적 무손실 비닝(categorical_features).
// LOB(max_lineage>0)는 조상 방향 상속해 고차 상호작용 근사. pybind11.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using u16 = uint16_t;

static inline double gain_term(double G, double H, double lam) {
  return G * G / (H + lam);
}

static double percentile_sorted(const std::vector<double>& xs, double p) {
  int n = (int)xs.size();
  if (n == 0) return 0.0;
  if (n == 1) return xs[0];
  double rank = p / 100.0 * (n - 1);
  int lo = (int)std::floor(rank);
  int hi = std::min(lo + 1, n - 1);
  return xs[lo] + (rank - lo) * (xs[hi] - xs[lo]);
}
static std::vector<double> unique_sorted(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

// ─── 전역 사전 binning (히스토그램 트릭의 핵심 - NaN 대응 개선) ──────────────
struct Bins {
  std::vector<u16> idx;                      // n*d bin index (row-major)
  std::vector<std::vector<double>> edges;    // [f] 정렬 경계
  std::vector<std::vector<double>> centers;  // [f] bin 대표값
  std::vector<bool> has_nan;                 // [f] 피처별 NaN 존재 여부
};

static Bins precompute_bins(const double* X, int n, int d, int max_bins,
                            const std::vector<int>& categorical) {
  Bins B;
  B.idx.resize((size_t)n * d);
  B.edges.resize(d);
  B.centers.resize(d);
  B.has_nan.assign(d, false);

  std::vector<double> col(n);
  for (int f = 0; f < d; f++) {
    std::vector<double> cs;
    cs.reserve(n);
    bool fnan = false;

    for (int i = 0; i < n; i++) {
      double val = X[(size_t)i * d + f];
      col[i] = val;
      if (std::isnan(val)) {
        fnan = true;
      } else {
        cs.push_back(val);
      }
    }
    B.has_nan[f] = fnan;

    if (cs.empty()) {
      B.edges[f] = {};
      B.centers[f] = {0.0};
      for (int i = 0; i < n; i++) B.idx[(size_t)i * d + f] = 0;
      continue;
    }

    std::sort(cs.begin(), cs.end());
    std::vector<double> e;
    bool is_cat = !categorical.empty() && categorical[f];
    if (is_cat) {
      // 범주형: 무손실 비닝 — 각 distinct 레벨이 자기 bin. 인접 unique 중점을 edge로.
      // max_bins(연속용 저해상도)에 영향받지 않아 고카디널리티 레벨이 병합되지 않음.
      std::vector<double> u = unique_sorted(cs);
      const int HARD_CAP = 8192;  // u16 bin index + 성능 상한
      if ((int)u.size() <= HARD_CAP) {
        for (size_t k = 0; k + 1 < u.size(); k++)
          e.push_back(0.5 * (u[k] + u[k + 1]));
      } else {  // 과도 카디널리티 → quantile 폴백
        for (int b = 1; b < HARD_CAP; b++)
          e.push_back(percentile_sorted(cs, 100.0 * b / HARD_CAP));
        e = unique_sorted(e);
      }
    } else {
      int actual_max_bins = fnan ? std::max(2, max_bins - 1) : max_bins;
      for (int b = 1; b < actual_max_bins; b++)
        e.push_back(percentile_sorted(cs, 100.0 * b / actual_max_bins));
      e = unique_sorted(e);
    }

    int norm_nb = (int)e.size() + 1;
    int total_nb = fnan ? norm_nb + 1 : norm_nb;
    u16 nan_bin_idx = (u16)(total_nb - 1);

    B.edges[f] = e;
    std::vector<double> sum(total_nb, 0.0);
    std::vector<int> cnt(total_nb, 0);

    for (int i = 0; i < n; i++) {
      if (std::isnan(col[i])) {
        B.idx[(size_t)i * d + f] = nan_bin_idx;
        cnt[nan_bin_idx]++;
      } else {
        int b = (int)(std::upper_bound(e.begin(), e.end(), col[i]) - e.begin());
        B.idx[(size_t)i * d + f] = (u16)b;
        sum[b] += col[i];
        cnt[b]++;
      }
    }

    std::vector<double>& ctr = B.centers[f];
    ctr.resize(total_nb);
    for (int b = 0; b < total_nb; b++) {
      if (b == nan_bin_idx && fnan) {
        ctr[b] = 0.0;
        continue;
      }
      if (cnt[b] > 0)
        ctr[b] = sum[b] / cnt[b];
      else
        ctr[b] = (b == 0 ? (e.empty() ? cs[0] : e[0])
                         : (b - 1 < (int)e.size() ? e[b - 1] : cs.back()));
    }
  }
  return B;
}

// ─── 분할/노드 ───────────────────────────────────────────────────────────────
struct Split {
  double gain = 0;
  int type = 0, fA = -1, fB = -1;
  double thr = 0, coefA = 0, coefB = 0, bias = 0;
  int nan_direction = 0;
};
struct Node {
  bool is_leaf = true;
  double weight = 0;
  int type = 0, fA = -1, fB = -1;
  double thr = 0, coefA = 0, coefB = 0, bias = 0;
  int left = -1, right = -1;
  int nan_direction = 0;
  double gain = 0;   // 채택 분할 gain (explain 경로 귀속용)
  int dir_id = -1;   // LOB: ≥0이면 dense 방향(dirs_[dir_id]) 기반 분할
  std::vector<double> leaf_weights;
};
struct Params {
  int n_estimators = 60, max_depth = 4, max_bins = 64, min_samples = 10;
  double learning_rate = 0.12, reg_lambda = 1.0;
  int n_screen = -1;
  double subsample = 1.0, colsample = 1.0;
  unsigned seed = 42;
  int objective = 0;
  int fast_dir = 1;  // 1="full" (all pairs, default), 2="fast" (Star anchor)
  // 회귀 손실: 0=squared(L2), 1=huber(robust), 2=quantile(pinball).
  int loss = 0;
  double alpha = 0.9;  // huber: |residual| delta 분위 / quantile: 목표 분위
  int clip = 0;        // 1=예측을 train 타깃 [min,max]로 clamp (외삽 폭주 방지)
  // 피처별 단조 제약: -1=감소, 0=무제약, +1=증가. 비면 제약 없음(fast path).
  std::vector<int> monotone;
  // 피처별 범주형 플래그(1=범주형 → 무손실 비닝). 비면 전부 연속.
  std::vector<int> categorical;
  // LOB(Lineage Oblique Boosting): >0이면 조상이 발견한 방향을 상속해 (z,raw),
  // (z,z) 쌍도 탐색 → 방향이 계층 합성. 0=off(기존 2D-pair). 값=lineage 최대 보유수.
  int max_lineage = 0;
};

// ─── 단일 히스토그램 해상도 (전역 사전 binning과 2D 투영 threshold 스캔이
// 공유). Booster::fit() 시작 시 P.max_bins로부터 한 번만 설정되고, 이후
// fit 동안은 읽기 전용이라 OpenMP 스레드 간 데이터 레이스 없음. 이전에는
// 2D 경로(refine_threshold)가 이 값을 무시하고 64를 하드코딩해서, 사용자가
// max_bins를 바꿔도 2D 분할 품질에는 영향이 없었음 — 이제 단일 소스로 통일. ──
static int g_hist_bins = 64;

// ─── 2D 계산용 재사용 Workspace 구조체 (thread_local 대체) ───────────────────
struct Workspace {
  std::vector<double> Gc, Hc, proj;
  std::vector<int> cnt;
  std::vector<int> oa, ob;
  std::vector<double> Gs, Hs;
  std::vector<int> so;
  std::vector<int> lab;
  std::vector<double> cA, cB;
  std::vector<double> histG,
      histH;  // refine_threshold 재사용 버퍼 (g_hist_bins 크기)
  std::vector<double> cols_flat;
  std::vector<u16> bins_flat;
  // eval_1d 재사용 버퍼
  std::vector<double> Ga, Ha;
  std::vector<int> occ;
  // refine_threshold lazy-reset dirty-bin 버퍼
  std::vector<int> dirty;
};

// ─── SIS 스크리닝 (캐시 프렌들리 Row-major 순회로 대폭 최적화)
// ────────────────
static std::vector<int> screen(const double* X, int d,
                               const std::vector<int>& idx,
                               const std::vector<double>& g, int m,
                               const std::vector<int>& candidates) {
  if (m < 0 || m >= (int)candidates.size()) {
    std::vector<int> a = candidates;
    std::sort(a.begin(), a.end());
    return a;
  }
  // m=0과 m=1을 동일하게 취급: 둘 다 "피처 1개만 통과"이므로 eval_2d가
  // pair를 만들 후보가 없어 자연히 단일 축(축 정렬) 분할로 회귀한다.
  m = std::max(1, m);
  int n = (int)idx.size();
  double gm = 0;
  for (int i = 0; i < n; i++) gm += g[idx[i]];
  gm /= n;
  double gvar = 0;
  for (int i = 0; i < n; i++) {
    double t = g[idx[i]] - gm;
    gvar += t * t;
  }
  double gstd = std::sqrt(gvar / n);
  std::vector<double> score(d, 0);

  if (gstd > 1e-12) {
    std::vector<double> xm(d, 0.0);
    std::vector<int> valid_count(d, 0);

    // Pass 1: 샘플(행)을 외곽 루프에 두어 메모리를 연속적으로 읽음 (캐시 최적화)
    for (int i = 0; i < n; i++) {
      size_t row_idx = (size_t)idx[i] * d;
      for (int f : candidates) {
        double v = X[row_idx + f];
        if (!std::isnan(v)) {
          xm[f] += v;
          valid_count[f]++;
        }
      }
    }

    for (int f : candidates) {
      if (valid_count[f] >= 2) xm[f] /= valid_count[f];
    }

    std::vector<double> xv(d, 0.0);
    std::vector<double> cov(d, 0.0);

    // Pass 2: 분산 및 공분산 계산도 동일하게 캐시 친화적 구조로 변경
    for (int i = 0; i < n; i++) {
      size_t row_idx = (size_t)idx[i] * d;
      double g_diff = g[idx[i]] - gm;
      for (int f : candidates) {
        if (valid_count[f] < 2) continue;
        double v = X[row_idx + f];
        if (!std::isnan(v)) {
          double xt = v - xm[f];
          xv[f] += xt * xt;
          cov[f] += xt * g_diff;
        }
      }
    }

    for (int f : candidates) {
      if (valid_count[f] < 2) {
        score[f] = 0;
        continue;
      }
      double xs = std::sqrt(xv[f] / valid_count[f]);
      score[f] =
          (xs > 1e-12) ? std::fabs(cov[f] / valid_count[f] / (xs * gstd)) : 0;
    }
  }
  std::vector<int> fs = candidates;
  std::partial_sort(fs.begin(), fs.begin() + m, fs.end(),
                    [&](int a, int b) { return score[a] > score[b]; });
  fs.resize(m);
  std::sort(fs.begin(), fs.end());
  return fs;
}

static std::vector<int> get_feature_candidates(int d, double colsample, std::mt19937& rng) {
  std::vector<int> candidates(d);
  std::iota(candidates.begin(), candidates.end(), 0);
  if (colsample < 1.0) {
    int keep = std::max(2, (int)std::ceil(colsample * d));
    std::shuffle(candidates.begin(), candidates.end(), rng);
    candidates.resize(keep);
    std::sort(candidates.begin(), candidates.end());
  }
  return candidates;
}

// ─── H-가중 LSQ 선형 분리면 (2×2) ───────────────────────────────────────────
static bool lsq_separator(const std::vector<double>& cA,
                          const std::vector<double>& cB,
                          const std::vector<int>& lab,
                          const std::vector<double>& Hs, double& oA,
                          double& oB) {
  int S = (int)cA.size();
  double a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
  for (int i = 0; i < S; i++) {
    double H = Hs[i], u = cA[i], v = cB[i], l = lab[i];
    a00 += H * u * u;
    a01 += H * u * v;
    a11 += H * v * v;
    b0 += H * u * l;
    b1 += H * v * l;
  }
  double det = a00 * a11 - a01 * a01, dA, dB;
  if (std::fabs(det) < 1e-10) {
    double w0 = 0, w1 = 0, m0u = 0, m0v = 0, m1u = 0, m1v = 0;
    for (int i = 0; i < S; i++) {
      if (lab[i] == 0) {
        w0 += Hs[i];
        m0u += Hs[i] * cA[i];
        m0v += Hs[i] * cB[i];
      } else {
        w1 += Hs[i];
        m1u += Hs[i] * cA[i];
        m1v += Hs[i] * cB[i];
      }
    }
    if (w0 < 1e-10 || w1 < 1e-10) return false;
    dA = m1u / w1 - m0u / w0;
    dB = m1v / w1 - m0v / w0;
  } else {
    dA = (a11 * b0 - a01 * b1) / det;
    dB = (-a01 * b0 + a00 * b1) / det;
  }
  double nrm = std::sqrt(dA * dA + dB * dB);
  if (nrm < 1e-10) return false;
  oA = dA / nrm;
  oB = dB / nrm;
  return true;
}

// ─── 투영 위 히스토그램 임계 (Fast Path 분리 및 분기 제거) ───────────────────
// Callers materialize `proj` and compute its valid-value range [mn, mx] in the
// same pass (NaN excluded); pass them in so we skip a redundant min/max scan.
// If there is no valid value the caller passes mn>mx (e.g. +inf/-inf) → reject.
static bool refine_threshold(const std::vector<double>& proj, const double* gn,
                             const double* hn, double lam, double Gp, double Hp,
                             double mn, double mx, double& outT, double& outGain,
                             bool has_nan, Workspace& ws) {
  int n = (int)proj.size();
  if (n == 0 || mx - mn < 1e-12) return false;

  const int B = g_hist_bins;
  double w = (mx - mn) / B;
  // B is small (= max_bins), so a flat zero of 2·B is cheaper than per-sample
  // dirty-bin bookkeeping; the inner loops stay branch-free (better vectorized).
  if ((int)ws.histG.size() < B) {
    ws.histG.assign(B, 0.0);
    ws.histH.assign(B, 0.0);
  }
  std::vector<double>& Gb = ws.histG;
  std::vector<double>& Hb = ws.histH;
  std::fill(Gb.begin(), Gb.begin() + B, 0.0);
  std::fill(Hb.begin(), Hb.begin() + B, 0.0);
  double G_nan = 0, H_nan = 0;

  if (!has_nan) {
    for (int i = 0; i < n; i++) {
      int b = (int)((proj[i] - mn) / w);
      if (b >= B) b = B - 1;
      if (b < 0) b = 0;
      Gb[b] += gn[i];
      Hb[b] += hn[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      if (std::isnan(proj[i])) {
        G_nan += gn[i];
        H_nan += hn[i];
      } else {
        int b = (int)((proj[i] - mn) / w);
        if (b >= B) b = B - 1;
        if (b < 0) b = 0;
        Gb[b] += gn[i];
        Hb[b] += hn[i];
      }
    }
  }

  double base = gain_term(Gp, Hp, lam), GL = 0, HL = 0, bg = 0, bt = 0;
  bool found = false;

  for (int b = 0; b + 1 < B; b++) {
    GL += Gb[b];
    HL += Hb[b];

    if (!has_nan) {
      if (HL <= 1e-12 || Hp - HL <= 1e-12) continue;
      double gain_val =
          gain_term(GL, HL, lam) + gain_term(Gp - GL, Hp - HL, lam) - base;
      if (gain_val > bg) {
        bg = gain_val;
        bt = mn + (b + 1) * w;
        found = true;
      }
    } else {
      for (int nan_dir = 0; nan_dir <= 1; nan_dir++) {
        double cur_GL = GL + (nan_dir == 0 ? G_nan : 0);
        double cur_HL = HL + (nan_dir == 0 ? H_nan : 0);
        if (cur_HL <= 1e-12 || Hp - cur_HL <= 1e-12) continue;
        double gain_val = gain_term(cur_GL, cur_HL, lam) +
                          gain_term(Gp - cur_GL, Hp - cur_HL, lam) - base;
        if (gain_val > bg) {
          bg = gain_val;
          bt = mn + (b + 1) * w;
          found = true;
        }
      }
    }
  }
  outT = bt;
  outGain = bg;
  return found;
}

struct FCache {
  int f;
  const double* col;
  const u16* bin;
};

static std::vector<FCache> build_caches(const double* X, int d,
                                        const std::vector<u16>& binidx,
                                        const std::vector<int>& idx,
                                        const std::vector<int>& feats,
                                        Workspace& ws) {
  size_t n_samples = idx.size();
  size_t n_feats = feats.size();
  ws.cols_flat.resize(n_feats * n_samples);
  ws.bins_flat.resize(n_feats * n_samples);

  std::vector<FCache> C(n_feats);
  for (size_t fi = 0; fi < n_feats; fi++) {
    C[fi].f = feats[fi];
    C[fi].col = ws.cols_flat.data() + fi * n_samples;
    C[fi].bin = ws.bins_flat.data() + fi * n_samples;
  }

  // 이 함수 역시 샘플(행)을 외곽 루프에 두어 캐시 적중률 극대화
  for (size_t i = 0; i < n_samples; i++) {
    size_t row_offset = (size_t)idx[i] * d;
    for (size_t fi = 0; fi < n_feats; fi++) {
      int f = C[fi].f;
      double* write_col = const_cast<double*>(C[fi].col);
      u16* write_bin = const_cast<u16*>(C[fi].bin);
      write_col[i] = X[row_offset + f];
      write_bin[i] = binidx[row_offset + f];
    }
  }
  return C;
}

// ─── 1D 분할 ─────────────────────────────────────────────────────────────────
static Split eval_1d(const std::vector<FCache>& C,
                     const std::vector<std::vector<double>>& centers,
                     const std::vector<double>& gn,
                     const std::vector<double>& hn, double Gp, double Hp,
                     const Params& P, Workspace& ws) {
  Split best;
  double base = gain_term(Gp, Hp, P.reg_lambda);
  int nloc = (int)gn.size();

  for (const FCache& c : C) {
    const std::vector<double>& ctr = centers[c.f];
    int k = (int)ctr.size();
    if (k < 2) continue;

    // Workspace 재사용: 매 피처마다 힙 할당 제거
    ws.Ga.assign(k, 0.0);
    ws.Ha.assign(k, 0.0);
    for (int i = 0; i < nloc; i++) {
      int b = c.bin[i];
      ws.Ga[b] += gn[i];
      ws.Ha[b] += hn[i];
    }

    ws.occ.clear();
    for (int a = 0; a < k; a++) {
      if (ws.Ha[a] > 0.0) ws.occ.push_back(a);  // h>0 ⟺ 점유
    }
    if ((int)ws.occ.size() < 2) continue;

    double GL = 0, HL = 0;
    for (int ki = 0; ki + 1 < (int)ws.occ.size(); ki++) {
      GL += ws.Ga[ws.occ[ki]];
      HL += ws.Ha[ws.occ[ki]];
      if (HL <= 1e-12 || (Hp - HL) <= 1e-12) continue;
      double gain_val = gain_term(GL, HL, P.reg_lambda) +
                        gain_term(Gp - GL, Hp - HL, P.reg_lambda) - base;
      if (gain_val > best.gain) {
        best.gain = gain_val;
        best.type = 1;
        best.fA = c.f;
        best.thr = (ctr[ws.occ[ki]] + ctr[ws.occ[ki + 1]]) / 2.0;
        best.nan_direction = 1;
      }
    }
  }
  return best;
}

// ─── 한 쌍 2D oblique (Workspace 주입 및 SIMD 고속 투영 적용) ────────────────
static Split eval_pair(const FCache& cA_, const FCache& cB_,
                       const std::vector<double>& ctrA,
                       const std::vector<double>& ctrB,
                       const std::vector<double>& gn,
                       const std::vector<double>& hn, double Gp, double Hp,
                       const Params& P, Workspace& ws, bool has_nan) {
  Split s;
  int kA = (int)ctrA.size(), kB = (int)ctrB.size();
  if (kA < 1 || kB < 1) return s;
  int nloc = (int)gn.size();
  double coefA, coefB;

  {
    // H-가중 gradient 직접 회귀: t=-g/h를 두 피처에 가중 LSQ → 방향.
    // 그리드 scatter·점유수집·정렬·BHC·LSQ 전부 생략, 9-스칼라 1패스 + 2×2.
    double Sh = 0, Sa = 0, Sb = 0, Saa = 0, Sab = 0, Sbb = 0, Sat = 0, Sbt = 0,
           St = 0;
    if (!has_nan) {
#if defined(__GNUC__) || defined(__clang__)
#pragma omp simd reduction(+:Sh,Sa,Sb,Saa,Sab,Sbb,Sat,Sbt,St)
#endif
      for (int i = 0; i < nloc; i++) {
        double xa = cA_.col[i], xb = cB_.col[i];
        double gi = gn[i], hi = hn[i];
        Sh += hi;
        Sa += hi * xa;
        Sb += hi * xb;
        Saa += hi * xa * xa;
        Sab += hi * xa * xb;
        Sbb += hi * xb * xb;
        Sat += -gi * xa;
        Sbt += -gi * xb;
        St += -gi;
      }
    } else {
      for (int i = 0; i < nloc; i++) {
        double xa = cA_.col[i], xb = cB_.col[i];
        if (std::isnan(xa) || std::isnan(xb)) continue;
        double gi = gn[i], hi = hn[i];
        Sh += hi;
        Sa += hi * xa;
        Sb += hi * xb;
        Saa += hi * xa * xa;
        Sab += hi * xa * xb;
        Sbb += hi * xb * xb;
        Sat += -gi * xa;
        Sbt += -gi * xb;
        St += -gi;
      }
    }
    if (Sh < 1e-12) return s;
    double A00 = Saa - Sa * Sa / Sh + P.reg_lambda, A01 = Sab - Sa * Sb / Sh,
           A11 = Sbb - Sb * Sb / Sh + P.reg_lambda;
    double b0 = Sat - Sa * St / Sh, b1 = Sbt - Sb * St / Sh;
    double det = A00 * A11 - A01 * A01;
    if (std::fabs(det) < 1e-12) return s;
    double dA = (A11 * b0 - A01 * b1) / det, dB = (A00 * b1 - A01 * b0) / det;
    double nrm = std::sqrt(dA * dA + dB * dB);
    if (nrm < 1e-12) return s;
    coefA = dA / nrm;
    coefB = dB / nrm;
  }

  // 단조 사분면 feasibility: 두 피처 모두 제약이고 방향이 충돌하면
  // (sign(coefA)·mA ≠ sign(coefB)·mB) 이 사선쌍으론 공동 단조 불가 → 기각
  // (eval_1d의 개별 축 분할이 폴백). 한쪽만 제약이면 방향 flip 자유로 항상 feasible.
  if (!P.monotone.empty()) {
    int mA = P.monotone[cA_.f], mB = P.monotone[cB_.f];
    if (mA && mB && std::fabs(coefA) > 1e-12 && std::fabs(coefB) > 1e-12) {
      int sA = coefA > 0 ? 1 : -1, sB = coefB > 0 ? 1 : -1;
      if (sA * mA != sB * mB) return s;  // gain=0 → 폴백
    }
  }

  ws.proj.resize(nloc);
  double pmn = std::numeric_limits<double>::infinity();
  double pmx = -std::numeric_limits<double>::infinity();
  if (!has_nan) {
// NaN 없으면 분기 없는 SIMD: 투영 + min/max reduction 한 패스로 융합
#if defined(__GNUC__) || defined(__clang__)
#pragma omp simd reduction(min : pmn) reduction(max : pmx)
#endif
    for (int i = 0; i < nloc; i++) {
      double p = coefA * cA_.col[i] + coefB * cB_.col[i];
      ws.proj[i] = p;
      pmn = p < pmn ? p : pmn;
      pmx = p > pmx ? p : pmx;
    }
  } else {
    for (int i = 0; i < nloc; i++) {
      if (std::isnan(cA_.col[i]) || std::isnan(cB_.col[i])) {
        ws.proj[i] = std::numeric_limits<double>::quiet_NaN();
      } else {
        double p = coefA * cA_.col[i] + coefB * cB_.col[i];
        ws.proj[i] = p;
        pmn = p < pmn ? p : pmn;
        pmx = p > pmx ? p : pmx;
      }
    }
  }
  double t, gn2;
  if (!refine_threshold(ws.proj, gn.data(), hn.data(), P.reg_lambda, Gp, Hp, pmn,
                        pmx, t, gn2, has_nan, ws))
    return s;
  s.gain = gn2;
  s.type = 2;
  s.fA = cA_.f;
  s.fB = cB_.f;
  s.coefA = coefA;
  s.coefB = coefB;
  s.bias = -t;
  s.nan_direction = 1;
  return s;
}

static Split eval_2d(const std::vector<FCache>& C,
                     const std::vector<std::vector<double>>& centers,
                     const std::vector<bool>& has_nan,
                     const std::vector<double>& gn,
                     const std::vector<double>& hn, double Gp, double Hp,
                     const Params& P, std::vector<Workspace>& wss) {
  int nf = (int)C.size();
  std::vector<std::pair<int, int>> pr;
  // fast_dir: 2="fast" (Star — anchor feat0 × rest, O(d)), else="full" (all pairs,
  // O(d²), accuracy). BHC(0)/top1(3) removed.
  if (P.fast_dir == 2) {
    if (nf > 1) {
      pr.reserve(nf - 1);
      for (int b = 1; b < nf; b++) pr.emplace_back(0, b);
    }
  } else {
    pr.reserve(nf * (nf - 1) / 2);
    for (int a = 0; a < nf; a++)
      for (int b = a + 1; b < nf; b++) pr.emplace_back(a, b);
  }
  int np = (int)pr.size();
  int nloc = (int)gn.size();

  int max_threads = 1;
#ifdef _OPENMP
  max_threads = omp_get_max_threads();
#endif
  // 작업량(쌍수×표본수)이 작으면 fork-join 오버헤드 > 이득 → serial 폴백.
  bool par = max_threads > 1 && (long)np * nloc > 30000;
  int req_threads = par ? max_threads : 1;
  if (wss.size() < (size_t)req_threads) {
    wss.resize(req_threads);
  }

  // res 벡터 힙 할당 제거: 스레드별 best를 직접 reduction으로 집계
  Split best;
#pragma omp parallel if (par)
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    Split local_best;
#pragma omp for schedule(static) nowait
    for (int p = 0; p < np; p++) {
      int fA = C[pr[p].first].f;
      int fB = C[pr[p].second].f;
      bool pair_has_nan = has_nan[fA] || has_nan[fB];
      Split s = eval_pair(C[pr[p].first], C[pr[p].second], centers[fA],
                          centers[fB], gn, hn, Gp, Hp, P, wss[tid], pair_has_nan);
      if (s.gain > local_best.gain) local_best = s;
    }
#pragma omp critical
    if (local_best.gain > best.gain) best = local_best;
  }
  return best;
}

// ─── 재귀 빌드 ───────────────────────────────────────────────────────────────
static int build(std::vector<Node>& arena, const double* X, int d,
                 const std::vector<u16>& binidx,
                 const std::vector<std::vector<double>>& centers,
                 const std::vector<bool>& has_nan, const std::vector<double>& g,
                 const std::vector<double>& h, std::vector<int> idx, int depth,
                 const Params& P, std::mt19937& rng, std::vector<double>& imp,
                 std::vector<double>& coef_imp, std::vector<double>& inter,
                 double lo, double hi, std::vector<Workspace>& wss, int K) {
  double Gp = 0, Hp = 0;
  bool is_mc = (P.objective == 2);
  if (!is_mc) {
    for (int i : idx) {
      Gp += g[i];
      Hp += h[i];
    }
  }
  int ni = (int)arena.size();
  arena.push_back(Node());
  if (is_mc) {
    arena[ni].weight = 0.0;
  } else {
    double w = -Gp / (Hp + P.reg_lambda);
    arena[ni].weight = w < lo ? lo : (w > hi ? hi : w);  // 단조 경계로 clamp
  }

  if (depth >= P.max_depth || (int)idx.size() < P.min_samples) {
    if (is_mc) {
      arena[ni].leaf_weights.resize(K, 0.0);
      for (int k = 0; k < K; k++) {
        double sum_g = 0.0, sum_h = 0.0;
        for (int i : idx) {
          sum_g += g[i * K + k];
          sum_h += h[i * K + k];
        }
        arena[ni].leaf_weights[k] = -sum_g / (sum_h + P.reg_lambda);
      }
    }
    return ni;
  }

  std::vector<double> gn(idx.size()), hn(idx.size());
  std::vector<double> gn_global;
  if (is_mc) {
    int max_idx = 0;
    for (int i : idx) if (i > max_idx) max_idx = i;
    gn_global.assign(max_idx + 1, 0.0);

    // Multiclass (joint softmax) reduces the K-dim gradient to a 1D contrast for
    // the oblique direction/threshold search; per-class leaf weights still update
    // all K. The contrast must be node-consistent and *signed* so samples split to
    // opposite sides — a per-sample magnitude has no sign and collapses the split.
    //
    // The softmax residual g_{i,k}=p_k-[y_i=k] sums to ~0 per sample, so the node
    // aggregate Σ_i g_{i,k} is most POSITIVE for the most over-predicted class and
    // most NEGATIVE for the most under-predicted one. Contrasting those two
    // opposite-sign extremes (k1,k2) gives the axis along which the node is
    // globally most miscalibrated — a genuinely discriminative direction.
    //
    // The previous heuristic picked the top-2 classes by Σ|g| (magnitude, sign-
    // blind): it could pick two same-sign (both over-predicted) classes, yielding
    // a near-degenerate contrast. Switching to signed extremes raises multiclass
    // accuracy and lowers logloss across digits/wine/synthetic K=3..10 suites.
    std::vector<double> SG(K, 0.0);
    for (int i : idx)
      for (int k = 0; k < K; k++) SG[k] += g[i * K + k];
    int k1 = 0, k2 = 0;
    for (int k = 1; k < K; k++) {
      if (SG[k] > SG[k1]) k1 = k;
      if (SG[k] < SG[k2]) k2 = k;
    }
    for (size_t i = 0; i < idx.size(); i++) {
      int idx_i = idx[i];
      double val_g = g[idx_i * K + k1] - g[idx_i * K + k2];
      double val_h = h[idx_i * K + k1] + h[idx_i * K + k2];
      gn[i] = val_g; hn[i] = val_h; gn_global[idx_i] = val_g;
      Gp += val_g; Hp += val_h;
    }
  } else {
    for (size_t i = 0; i < idx.size(); i++) {
      gn[i] = g[idx[i]];
      hn[i] = h[idx[i]];
    }
  }

  std::vector<int> candidates = get_feature_candidates(d, P.colsample, rng);
  auto feats = screen(X, d, idx, (is_mc ? gn_global : g), P.n_screen, candidates);
  if (wss.empty()) wss.resize(1);
  auto C = build_caches(X, d, binidx, idx, feats, wss[0]);
  // 2D 사선이 주 분할.
  Split bs = eval_2d(C, centers, has_nan, gn, hn, Gp, Hp, P, wss);
  if (bs.gain <= 1e-6 || bs.type == 0)
    bs = eval_1d(C, centers, gn, hn, Gp, Hp, P, wss[0]);
  if (bs.gain <= 1e-6 || bs.type == 0) {
    if (is_mc) {
      arena[ni].leaf_weights.resize(K, 0.0);
      for (int k = 0; k < K; k++) {
        double sum_g = 0.0, sum_h = 0.0;
        for (int i : idx) {
          sum_g += g[i * K + k];
          sum_h += h[i * K + k];
        }
        arena[ni].leaf_weights[k] = -sum_g / (sum_h + P.reg_lambda);
      }
    }
    return ni;
  }

  std::vector<int> li, ri;
  for (int i : idx) {
    bool left;
    if (bs.type == 1) {
      double val = X[(size_t)i * d + bs.fA];
      if (std::isnan(val))
        left = (bs.nan_direction == 0);
      else
        left = val < bs.thr;
    } else {
      double valA = X[(size_t)i * d + bs.fA];
      double valB = X[(size_t)i * d + bs.fB];
      if (std::isnan(valA) || std::isnan(valB)) {
        left = (bs.nan_direction == 0);
      } else {
        double sc = bs.coefA * valA + bs.coefB * valB + bs.bias;
        left = sc < 0;
      }
    }
    (left ? li : ri).push_back(i);
  }
  if (li.empty() || ri.empty()) return ni;
  // feature importance: 채택된 분할의 gain을 참여 피처에 누적
  imp[bs.fA] += bs.gain;
  if (bs.type == 2) imp[bs.fB] += bs.gain;
  // OQBoost 네이티브 설명: 계수 가중 importance + 사선쌍 interaction.
  if (bs.type == 1) {
    coef_imp[bs.fA] += bs.gain;  // 1D는 coef=1
  } else {
    double aA = std::fabs(bs.coefA), aB = std::fabs(bs.coefB);
    coef_imp[bs.fA] += bs.gain * aA;
    coef_imp[bs.fB] += bs.gain * aB;
    inter[(size_t)bs.fA * d + bs.fB] += bs.gain * aA * aB;  // 상삼각(fA<fB)
  }
  arena[ni].gain = bs.gain;
  arena[ni].is_leaf = false;
  arena[ni].type = bs.type;
  arena[ni].fA = bs.fA;
  arena[ni].fB = bs.fB;
  arena[ni].thr = bs.thr;
  arena[ni].coefA = bs.coefA;
  arena[ni].coefB = bs.coefB;
  arena[ni].bias = bs.bias;
  arena[ni].nan_direction = bs.nan_direction;

  // 단조 경계 전파: 제약 피처가 분할에 관여하면 자식 출력 범위를 중점 m으로
  // 분리 → "위쪽" 자식은 [m,hi], "아래쪽"은 [lo,m]. 깊은 서브트리까지 leaf clamp가
  // 전역 단조를 보장(고정 타 피처 직선 위에선 사선분할도 단조 feature의 단일 threshold).
  double lo_l = lo, hi_l = hi, lo_r = lo, hi_r = hi;
  if (!P.monotone.empty()) {
    bool constrained = false, up_is_right = false;
    if (bs.type == 1) {
      int mA = P.monotone[bs.fA];
      if (mA) { constrained = true; up_is_right = (mA > 0); }  // x≥thr=right=high-fA
    } else {
      int mA = P.monotone[bs.fA], mB = P.monotone[bs.fB];
      if (mA && std::fabs(bs.coefA) > 1e-12) {
        constrained = true; up_is_right = ((bs.coefA > 0) == (mA > 0));
      } else if (mB && std::fabs(bs.coefB) > 1e-12) {
        constrained = true; up_is_right = ((bs.coefB > 0) == (mB > 0));
      }
    }
    if (constrained) {
      double GL = 0, HL = 0;
      for (int i : li) { GL += g[i]; HL += h[i]; }
      double wL = -GL / (HL + P.reg_lambda);
      double wR = -(Gp - GL) / ((Hp - HL) + P.reg_lambda);
      double m = 0.5 * (wL + wR);
      m = m < lo ? lo : (m > hi ? hi : m);  // [lo,hi]로 clamp → 자식 범위 항상 유효
      if (up_is_right) { hi_l = m; lo_r = m; }
      else { lo_l = m; hi_r = m; }
    }
  }

  int L = build(arena, X, d, binidx, centers, has_nan, g, h, std::move(li),
                depth + 1, P, rng, imp, coef_imp, inter, lo_l, hi_l, wss, K);
  int R = build(arena, X, d, binidx, centers, has_nan, g, h, std::move(ri),
                depth + 1, P, rng, imp, coef_imp, inter, lo_r, hi_r, wss, K);
  arena[ni].left = L;
  arena[ni].right = R;
  return ni;
}

// LOB dense 방향 분할의 child 선택. s = dot(dir,x)+bias, s<0 → left(0).
// NaN 관련 피처(coef≠0)가 있으면 nan_direction. (coef=0 피처의 NaN은 무시.)
static inline int lob_child(const Node& nd,
                            const std::vector<std::vector<double>>& dirs,
                            const double* x) {
  const std::vector<double>& c = dirs[nd.dir_id];
  double s = nd.bias;
  for (size_t k = 0; k < c.size(); k++) {
    if (c[k] != 0.0) {
      double v = x[k];
      if (std::isnan(v)) return nd.nan_direction;
      s += c[k] * v;
    }
  }
  return (s < 0) ? 0 : 1;
}

static inline double predict_one(const std::vector<Node>& A, const double* x,
                                 const std::vector<std::vector<double>>& dirs) {
  int ni = 0;
  while (true) {
    const Node& nd = A[ni];
    if (nd.is_leaf) return nd.weight;
    int ch;
    if (nd.dir_id >= 0) {
      ch = lob_child(nd, dirs, x);
    } else if (nd.type == 1) {
      ch = std::isnan(x[nd.fA]) ? nd.nan_direction : (x[nd.fA] < nd.thr ? 0 : 1);
    } else if (std::isnan(x[nd.fA]) || std::isnan(x[nd.fB])) {
      ch = nd.nan_direction;
    } else {
      double s = nd.coefA * x[nd.fA] + nd.coefB * x[nd.fB] + nd.bias;
      ch = (s < 0) ? 0 : 1;
    }
    ni = (ch == 0) ? nd.left : nd.right;
  }
}

// 표본이 도달하는 leaf의 arena 인덱스 (leaf-value line-search용).
static inline int leaf_index(const std::vector<Node>& A, const double* x,
                             const std::vector<std::vector<double>>& dirs) {
  int ni = 0;
  while (true) {
    const Node& nd = A[ni];
    if (nd.is_leaf) return ni;
    int ch;
    if (nd.dir_id >= 0) {
      ch = lob_child(nd, dirs, x);
    } else if (nd.type == 1) {
      ch = std::isnan(x[nd.fA]) ? nd.nan_direction : (x[nd.fA] < nd.thr ? 0 : 1);
    } else if (std::isnan(x[nd.fA]) || std::isnan(x[nd.fB])) {
      ch = nd.nan_direction;
    } else {
      double s = nd.coefA * x[nd.fA] + nd.coefB * x[nd.fB] + nd.bias;
      ch = (s < 0) ? 0 : 1;
    }
    ni = (ch == 0) ? nd.left : nd.right;
  }
}

// LOB: 두 투영 U,V에 H-가중 gradient 회귀 2×2 → (ca,cb) 비정규화 방향.
static bool solve_dir2x2(const double* U, const double* V, const double* gn,
                         const double* hn, int n, double lam,
                         double& ca, double& cb) {
  double Sh = 0, Sa = 0, Sb = 0, Saa = 0, Sab = 0, Sbb = 0, Sat = 0, Sbt = 0, St = 0;
  for (int i = 0; i < n; i++) {
    double u = U[i], v = V[i];
    if (std::isnan(u) || std::isnan(v)) continue;
    double gi = gn[i], hi = hn[i];
    Sh += hi; Sa += hi * u; Sb += hi * v;
    Saa += hi * u * u; Sab += hi * u * v; Sbb += hi * v * v;
    Sat += -gi * u; Sbt += -gi * v; St += -gi;
  }
  if (Sh < 1e-12) return false;
  double A00 = Saa - Sa * Sa / Sh + lam, A01 = Sab - Sa * Sb / Sh,
         A11 = Sbb - Sb * Sb / Sh + lam;
  double b0 = Sat - Sa * St / Sh, b1 = Sbt - Sb * St / Sh;
  double det = A00 * A11 - A01 * A01;
  if (std::fabs(det) < 1e-12) return false;
  ca = (A11 * b0 - A01 * b1) / det;
  cb = (A00 * b1 - A01 * b0) / det;
  return true;
}

// 가중 없는 분위수 (in-place nth_element).
static inline double quantile_inplace(std::vector<double>& v, double q) {
  if (v.empty()) return 0.0;
  size_t k = (size_t)std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// ─── Booster ─────────────────────────────────────────────────────────────────
class Booster {
 public:
  Params P;
  std::vector<std::vector<Node>> trees;
  double init_score = 0;
  double y_lo = 0, y_hi = 0;  // clip용 train 타깃 범위
  int d_ = 0;                        // 피처 수 (interaction reshape용)
  int K_ = 1;                        // 클래스 수
  std::vector<double> init_mc_;      // objective==2: 클래스별 log-prior 베이스라인
  std::vector<double> importances_;  // 피처별 누적 gain
  std::vector<double> coef_imp_;     // Σ gain·|coef| (계수 가중 importance)
  std::vector<double> inter_;        // d×d 상삼각: Σ gain·|a|·|b| (사선쌍 interaction)
  std::vector<std::vector<double>> dirs_;  // LOB dense 방향 테이블 (node.dir_id 참조)
  std::vector<double> w_;            // sample_weight (empty = unweighted)
  // early stopping: monitor a validation set, stop after `es_patience_` rounds
  // without > es_tol_ improvement in the deviance, truncate to the best round.
  std::vector<double> xval_, yval_;
  int nval_ = 0, es_patience_ = 0;
  double es_tol_ = 1e-4;
  int best_iteration_ = -1;
  Booster(int n_estimators, double learning_rate, int max_depth, int max_bins,
          double reg_lambda, int min_samples, int n_screen, double subsample,
          double colsample, unsigned seed, int objective, int fast_dir,
          int loss, double alpha, int clip, std::vector<int> monotone,
          std::vector<int> categorical, int max_lineage) {
    P.n_estimators = n_estimators;
    P.learning_rate = learning_rate;
    P.max_depth = max_depth;
    P.max_bins = max_bins;
    P.reg_lambda = reg_lambda;
    P.min_samples = min_samples;
    P.n_screen = n_screen;
    P.subsample = subsample;
    P.colsample = colsample;
    P.seed = seed;
    P.objective = objective;
    P.fast_dir = fast_dir;
    P.loss = loss;
    P.alpha = alpha;
    P.clip = clip;
    P.monotone = std::move(monotone);
    P.categorical = std::move(categorical);
    P.max_lineage = max_lineage;
  }

  // sample_weight: empty array = unweighted. In Newton boosting, multiplying g
  // and h by w applies the weight exactly (leaf=-G/H, gain, direction-fit all use
  // the weighted g/h).
  void set_weights(py::array_t<double, py::array::c_style | py::array::forcecast> wa, int n) {
    auto wb = wa.request();
    if ((int)wb.size == n) {
      const double* wp = (const double*)wb.ptr;
      w_.assign(wp, wp + n);
    } else {
      w_.clear();
    }
  }

  // validation set for early stopping (empty = disabled).
  void set_eval(py::array_t<double, py::array::c_style | py::array::forcecast> Xva,
                py::array_t<double, py::array::c_style | py::array::forcecast> yva,
                int patience, double tol) {
    auto xb = Xva.request();
    auto yb = yva.request();
    nval_ = (int)yb.size;
    if (nval_ > 0) {
      const double* xp = (const double*)xb.ptr;
      const double* yp = (const double*)yb.ptr;
      xval_.assign(xp, xp + xb.size);
      yval_.assign(yp, yp + nval_);
      es_patience_ = patience;
      es_tol_ = tol;
    } else {
      xval_.clear(); yval_.clear(); es_patience_ = 0;
    }
  }

  void fit(py::array_t<double, py::array::c_style | py::array::forcecast> Xa,
           py::array_t<double, py::array::c_style | py::array::forcecast> ya,
           py::array_t<double, py::array::c_style | py::array::forcecast> wa,
           py::array_t<double, py::array::c_style | py::array::forcecast> Xva,
           py::array_t<double, py::array::c_style | py::array::forcecast> yva,
           int es_patience, double es_tol) {
    auto Xb = Xa.request();
    auto yb = ya.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    const double* y = (const double*)yb.ptr;
    set_weights(wa, n);
    set_eval(Xva, yva, es_patience, es_tol);
    best_iteration_ = -1;
    // 단일 소스로 통일: max_bins가 전역 사전 binning과 2D threshold 스캔
    // 해상도를 동시에 결정. fit() 호출당 한 번만 쓰고 이후 읽기 전용.
    g_hist_bins = std::max(2, P.max_bins);
    Bins B = precompute_bins(X, n, d, P.max_bins, P.categorical);

    bool wt = !w_.empty();
    double sw = 0, ybar = 0;
    for (int i = 0; i < n; i++) { double wi = wt ? w_[i] : 1.0; ybar += wi * y[i]; sw += wi; }
    ybar /= (sw > 0 ? sw : 1.0);
    y_lo = y_hi = (n ? y[0] : 0.0);
    for (int i = 0; i < n; i++) { y_lo = std::min(y_lo, y[i]); y_hi = std::max(y_hi, y[i]); }
    if (P.objective == 2) {
      K_ = (int)y_hi + 1;
      init_score = 0.0;
      // Per-class log-prior baseline (softmax is shift-invariant, so log p_k is
      // enough). Binary inits to logit(prior); without this multiclass starts from
      // a uniform 1/K and must spend early trees just rebuilding class priors —
      // worst on imbalanced targets. Counts are sample-weighted so class_weight
      // shifts the baseline toward up-weighted classes (the intended correction).
      init_mc_.assign(K_, 0.0);
      std::vector<double> cnt(K_, 0.0);
      double tot = 0.0;
      for (int i = 0; i < n; i++) {
        double wi = wt ? w_[i] : 1.0;
        int yi = (int)y[i];
        if (yi >= 0 && yi < K_) { cnt[yi] += wi; tot += wi; }
      }
      const double eps = 1e-3;
      for (int k = 0; k < K_; k++)
        init_mc_[k] = std::log((cnt[k] + eps) / (tot + eps * K_));
    } else if (P.objective == 0) {
      K_ = 1;
      double y2 = std::min(std::max(ybar, 1e-6), 1 - 1e-6);
      init_score = std::log(y2 / (1 - y2));
    } else if (P.loss == 0) {
      K_ = 1;
      init_score = ybar;  // squared -> weighted mean
    } else {
      K_ = 1;
      // huber/quantile -> median (unweighted; init only, minor effect)
      std::vector<double> ys(y, y + n);
      size_t mid = ys.size() / 2;
      std::nth_element(ys.begin(), ys.begin() + mid, ys.end());
      init_score = ys[mid];
    }
    std::vector<double> raw(P.objective == 2 ? n * K_ : n, init_score);
    if (P.objective == 2)
      for (int i = 0; i < n; i++)
        for (int k = 0; k < K_; k++) raw[i * K_ + k] = init_mc_[k];
    trees.clear();
    trees.reserve(P.n_estimators);
    d_ = d;
    importances_.assign(d, 0.0);
    coef_imp_.assign(d, 0.0);
    inter_.assign((size_t)d * d, 0.0);
    dirs_.clear();
    rng_.seed(P.seed);
    boost_rounds(X, y, n, d, B, raw, P.n_estimators);
  }

  // warm-start: 기존 트리 위에 extra 라운드 추가. raw 상태는 저장된 트리로
  // 재구성하므로 별도 상태 보존 불필요. init_score·importances·rng_ 이어서 사용.
  void fit_more(py::array_t<double, py::array::c_style | py::array::forcecast> Xa,
                py::array_t<double, py::array::c_style | py::array::forcecast> ya,
                int extra,
                py::array_t<double, py::array::c_style | py::array::forcecast> wa) {
    auto Xb = Xa.request();
    auto yb = ya.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    const double* y = (const double*)yb.ptr;
    if (extra <= 0 || trees.empty()) return;
    set_weights(wa, n);
    es_patience_ = 0; nval_ = 0;  // no early stopping on warm-start continuation
    g_hist_bins = std::max(2, P.max_bins);
    Bins B = precompute_bins(X, n, d, P.max_bins, P.categorical);
    bool is_mc = (P.objective == 2);
    std::vector<double> raw(is_mc ? n * K_ : n, is_mc ? 0.0 : init_score);
    accumulate_raw(X, n, d, raw.data());
    if ((int)importances_.size() != d) importances_.assign(d, 0.0);
    trees.reserve(trees.size() + extra);
    boost_rounds(X, y, n, d, B, raw, extra);
  }

  int n_trees() const { return (int)trees.size(); }

  // deviance from incrementally-maintained validation raw scores (lower better):
  // logloss for classification, MSE for regression.
  double deviance_from(const std::vector<double>& raw_val) {
    double s = 0;
    if (P.objective == 2) {
      int K = K_;
      for (int j = 0; j < nval_; j++) {
        double max_val = raw_val[j * K];
        for (int k = 1; k < K; k++) {
          if (raw_val[j * K + k] > max_val) max_val = raw_val[j * K + k];
        }
        double sum_exp = 0.0;
        for (int k = 0; k < K; k++) {
          sum_exp += std::exp(raw_val[j * K + k] - max_val);
        }
        int yj = (int)yval_[j];
        double pyj = std::exp(raw_val[j * K + yj] - max_val) / sum_exp;
        pyj = std::min(std::max(pyj, 1e-12), 1 - 1e-12);
        s += -std::log(pyj);
      }
    } else {
      for (int j = 0; j < nval_; j++) {
        if (P.objective == 0) {
          double p = 1.0 / (1.0 + std::exp(-raw_val[j]));
          p = std::min(std::max(p, 1e-12), 1 - 1e-12);
          s += -(yval_[j] * std::log(p) + (1 - yval_[j]) * std::log(1 - p));
        } else {
          double e = raw_val[j] - yval_[j];
          s += e * e;
        }
      }
    }
    return s / std::max(1, nval_);
  }

  // boosting round runner (shared by fit / fit_more). Updates rng_, trees,
  // importances_, raw; honours early stopping when a validation set is set.
  void boost_rounds(const double* X, const double* y, int n, int d,
                    const Bins& B, std::vector<double>& raw, int rounds) {
    bool es = es_patience_ > 0 && nval_ > 0;
    double best_dev = std::numeric_limits<double>::infinity();
    int best_round = -1, since_best = 0;
    std::vector<double> raw_val(es ? (P.objective == 2 ? nval_ * K_ : nval_) : 0, init_score);
    if (es && P.objective == 2 && (int)init_mc_.size() == K_)
      for (int j = 0; j < nval_; j++)
        for (int k = 0; k < K_; k++) raw_val[j * K_ + k] = init_mc_[k];
    std::vector<Workspace> wss;
    if (es) {
      if (P.objective == 2) {
        int K = K_;
        for (int j = 0; j < nval_; j++) {
          for (auto& tr : trees) {
            int li = leaf_index(tr, xval_.data() + (size_t)j * d, dirs_);
            const auto& nd = tr[li];
            if ((int)nd.leaf_weights.size() == K) {
              for (int k = 0; k < K; k++) {
                raw_val[j * K + k] += P.learning_rate * nd.leaf_weights[k];
              }
            }
          }
        }
      } else {
        for (int j = 0; j < nval_; j++) {
          for (auto& tr : trees) {
            raw_val[j] += P.learning_rate * predict_one(tr, xval_.data() + (size_t)j * d, dirs_);
          }
        }
      }
    }
    std::vector<double> g(P.objective == 2 ? n * K_ : n), h(P.objective == 2 ? n * K_ : n), absr;
    if (P.objective == 1 && P.loss == 1) absr.resize(n);  // huber delta 계산용
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    int n_sub = std::max(1, (int)(P.subsample * n));

    for (int t = 0; t < rounds; t++) {
      if (P.objective == 2) {
        int K = K_;
#pragma omp parallel for if (n * K > 4000)
        for (int i = 0; i < n; i++) {
          double max_val = raw[i * K];
          for (int k = 1; k < K; k++) {
            if (raw[i * K + k] > max_val) max_val = raw[i * K + k];
          }
          double sum_exp = 0.0;
          std::vector<double> p(K);
          for (int k = 0; k < K; k++) {
            p[k] = std::exp(raw[i * K + k] - max_val);
            sum_exp += p[k];
          }
          int yi = (int)y[i];
          for (int k = 0; k < K; k++) {
            double pk = p[k] / sum_exp;
            g[i * K + k] = pk - (yi == k ? 1.0 : 0.0);
            h[i * K + k] = pk * (1.0 - pk);
          }
        }
      } else if (P.objective == 0) {
#pragma omp parallel for if (n > 8000)
        for (int i = 0; i < n; i++) {
          double p = 1.0 / (1.0 + std::exp(-raw[i]));
          g[i] = p - y[i];
          h[i] = p * (1 - p);
        }
      } else if (P.loss == 0) {  // squared (L2)
#pragma omp parallel for if (n > 8000)
        for (int i = 0; i < n; i++) {
          g[i] = raw[i] - y[i];
          h[i] = 1.0;
        }
      } else if (P.loss == 1) {  // huber — delta 안은 L2, 밖은 clip된 상수 gradient
        for (int i = 0; i < n; i++) absr[i] = std::fabs(raw[i] - y[i]);
        size_t k = (size_t)(P.alpha * (n - 1));
        std::nth_element(absr.begin(), absr.begin() + k, absr.end());
        double delta = absr[k];
        if (delta <= 0) delta = 1e-12;
#pragma omp parallel for if (n > 8000)
        for (int i = 0; i < n; i++) {
          double r = raw[i] - y[i];
          g[i] = r > delta ? delta : (r < -delta ? -delta : r);
          h[i] = 1.0;
        }
      } else {  // quantile (pinball), 목표 분위 alpha
#pragma omp parallel for if (n > 8000)
        for (int i = 0; i < n; i++) {
          double r = y[i] - raw[i];
          g[i] = r > 0 ? -P.alpha : (1.0 - P.alpha);
          h[i] = 1.0;
        }
      }
      if (!w_.empty()) {  // sample_weight: scale g,h by w (exact Newton weighting)
        if (P.objective == 2) {
          int K = K_;
#pragma omp parallel for if (n * K > 4000)
          for (int i = 0; i < n; i++) {
            double wi = w_[i];
            for (int k = 0; k < K; k++) {
              g[i * K + k] *= wi;
              h[i * K + k] *= wi;
            }
          }
        } else {
#pragma omp parallel for if (n > 8000)
          for (int i = 0; i < n; i++) { g[i] *= w_[i]; h[i] *= w_[i]; }
        }
      }

      std::vector<int> rows;
      if (P.subsample < 1.0) {
        std::shuffle(all.begin(), all.end(), rng_);
        rows.assign(all.begin(), all.begin() + n_sub);
      } else
        rows = all;
      std::vector<Node> arena;
      arena.reserve(256);
      if (P.max_lineage > 0)
        build_lob(arena, X, d, g, h, rows, {}, 0,
                  -std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity());
      else
        build(arena, X, d, B.idx, B.centers, B.has_nan, g, h, rows, 0, P, rng_,
              importances_, coef_imp_, inter_,
              -std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity(), wss, K_);

      // quantile: gradient는 부호뿐(±α)이라 -G/H leaf 값이 무의미 → 트리 구조는
      // gradient로 만들되 leaf 값을 멤버 residual의 alpha-분위로 line-search.
      if (P.objective == 1 && P.loss == 2) {
        std::vector<std::vector<double>> res(arena.size());
        for (int i : rows)
          res[leaf_index(arena, X + (size_t)i * d, dirs_)].push_back(y[i] - raw[i]);
        for (size_t li = 0; li < arena.size(); li++)
          if (arena[li].is_leaf && !res[li].empty())
            arena[li].weight = quantile_inplace(res[li], P.alpha);
      }

      if (P.objective == 2) {
        int K = K_;
#pragma omp parallel for if (n > 2000)
        for (int i = 0; i < n; i++) {
          int li = leaf_index(arena, X + (size_t)i * d, dirs_);
          const auto& nd = arena[li];
          if ((int)nd.leaf_weights.size() == K) {
            for (int k = 0; k < K; k++) {
              raw[i * K + k] += P.learning_rate * nd.leaf_weights[k];
            }
          }
        }
      } else {
#pragma omp parallel for if (n > 2000)
        for (int i = 0; i < n; i++)
          raw[i] += P.learning_rate * predict_one(arena, X + (size_t)i * d, dirs_);
      }
      trees.push_back(std::move(arena));

      if (es) {  // early stopping: update val scores, track best, stop on patience
        const std::vector<Node>& tr = trees.back();
        if (P.objective == 2) {
          int K = K_;
          for (int j = 0; j < nval_; j++) {
            int li = leaf_index(tr, xval_.data() + (size_t)j * d, dirs_);
            const auto& nd = tr[li];
            if ((int)nd.leaf_weights.size() == K) {
              for (int k = 0; k < K; k++) {
                raw_val[j * K + k] += P.learning_rate * nd.leaf_weights[k];
              }
            }
          }
        } else {
          for (int j = 0; j < nval_; j++)
            raw_val[j] += P.learning_rate *
                          predict_one(tr, xval_.data() + (size_t)j * d, dirs_);
        }
        double dev = deviance_from(raw_val);
        if (dev < best_dev - es_tol_) {
          best_dev = dev; best_round = (int)trees.size() - 1; since_best = 0;
        } else if (++since_best >= es_patience_) {
          break;
        }
      }
    }
    if (es) {  // keep only up to the best round
      if (best_round >= 0 && best_round + 1 < (int)trees.size())
        trees.resize(best_round + 1);
      best_iteration_ = (int)trees.size() - 1;
    }
  }

  // ─── LOB 빌드: 조상 방향(lineage)을 상속해 (raw,raw)/(z,raw)/(z,z) 2D 탐색.
  // 각 분할은 여전히 2×2. 채택 방향은 dense coef로 합성되어 dirs_에 저장되고
  // 자식 lineage로 전달(최대 P.max_lineage개). root=전수, 깊은노드=SIS 스크리닝.
  int build_lob(std::vector<Node>& arena, const double* X, int d,
                const std::vector<double>& g, const std::vector<double>& h,
                std::vector<int> idx, std::vector<std::vector<double>> lineage,
                int depth, double lo, double hi) {
    double Gp = 0, Hp = 0;
    for (int i : idx) { Gp += g[i]; Hp += h[i]; }
    int ni = (int)arena.size();
    arena.push_back(Node());
    double w = -Gp / (Hp + P.reg_lambda);
    arena[ni].weight = w < lo ? lo : (w > hi ? hi : w);
    int nloc = (int)idx.size();
    if (depth >= P.max_depth || nloc < P.min_samples) return ni;

    std::vector<double> gn(nloc), hn(nloc);
    for (int t = 0; t < nloc; t++) { gn[t] = g[idx[t]]; hn[t] = h[idx[t]]; }

    std::vector<int> feats;
    if (depth == 0) { feats.resize(d); std::iota(feats.begin(), feats.end(), 0); }
    else {
      std::vector<int> candidates = get_feature_candidates(d, P.colsample, rng_);
      feats = screen(X, d, idx, g, P.n_screen, candidates);
    }

    int K = (int)lineage.size();
    std::vector<std::vector<double>> zproj(K, std::vector<double>(nloc));
    for (int k = 0; k < K; k++)
      for (int t = 0; t < nloc; t++) {
        const double* x = X + (size_t)idx[t] * d;
        double s = 0; const std::vector<double>& c = lineage[k];
        for (int j = 0; j < d; j++) if (c[j] != 0.0) s += c[j] * x[j];
        zproj[k][t] = s;
      }
    // 후보 소스: (0,raw_f) 또는 (1,lineage_k)
    std::vector<std::pair<int, int>> src;
    for (int f : feats) src.push_back({0, f});
    for (int k = 0; k < K; k++) src.push_back({1, k});
    int S = (int)src.size();

    std::vector<double> U(nloc), V(nloc), proj(nloc);
    auto fill = [&](std::pair<int, int> s, std::vector<double>& out) {
      if (s.first == 0) for (int t = 0; t < nloc; t++) out[t] = X[(size_t)idx[t] * d + s.second];
      else out = zproj[s.second];
    };
    auto coefof = [&](std::pair<int, int> s) {
      std::vector<double> c(d, 0.0);
      if (s.first == 0) c[s.second] = 1.0; else c = lineage[s.second];
      return c;
    };
    Workspace ws;
    double bestg = 0, best_t = 0; std::vector<double> best_cz, best_proj;
    for (int a = 0; a < S; a++) {
      fill(src[a], U);
      for (int b = a + 1; b < S; b++) {
        fill(src[b], V);
        double ca, cb;
        if (!solve_dir2x2(U.data(), V.data(), gn.data(), hn.data(), nloc, P.reg_lambda, ca, cb))
          continue;
        std::vector<double> cu = coefof(src[a]), cv = coefof(src[b]);
        std::vector<double> cz(d); double nrm = 0;
        for (int j = 0; j < d; j++) { cz[j] = ca * cu[j] + cb * cv[j]; nrm += cz[j] * cz[j]; }
        nrm = std::sqrt(nrm);
        if (nrm < 1e-12) continue;
        bool hasnan = false;
        double pmn = std::numeric_limits<double>::infinity();
        double pmx = -std::numeric_limits<double>::infinity();
        for (int t = 0; t < nloc; t++) {
          double p = (ca * U[t] + cb * V[t]) / nrm;
          proj[t] = p;
          if (std::isnan(p)) hasnan = true;
          else { pmn = p < pmn ? p : pmn; pmx = p > pmx ? p : pmx; }
        }
        double t_thr, gn2;
        if (!refine_threshold(proj, gn.data(), hn.data(), P.reg_lambda, Gp, Hp,
                              pmn, pmx, t_thr, gn2, hasnan, ws))
          continue;
        if (gn2 > bestg) {
          for (int j = 0; j < d; j++) cz[j] /= nrm;
          bestg = gn2; best_t = t_thr; best_cz = cz; best_proj = proj;
        }
      }
    }
    if (bestg <= 1e-6) return ni;
    std::vector<int> li, ri;
    for (int t = 0; t < nloc; t++) {
      bool left = std::isnan(best_proj[t]) ? false : (best_proj[t] < best_t);
      (left ? li : ri).push_back(idx[t]);
    }
    if (li.empty() || ri.empty()) return ni;
    for (int j = 0; j < d; j++)
      if (best_cz[j] != 0.0) {
        double c = bestg * std::fabs(best_cz[j]);
        importances_[j] += c; coef_imp_[j] += c;
      }
    int did = (int)dirs_.size(); dirs_.push_back(best_cz);
    arena[ni].dir_id = did; arena[ni].is_leaf = false;
    arena[ni].gain = bestg; arena[ni].bias = -best_t; arena[ni].nan_direction = 1;
    std::vector<std::vector<double>> clin = lineage;
    clin.push_back(best_cz);
    if ((int)clin.size() > P.max_lineage)
      clin.erase(clin.begin(), clin.begin() + (clin.size() - P.max_lineage));
    int L = build_lob(arena, X, d, g, h, std::move(li), clin, depth + 1, lo, hi);
    int R = build_lob(arena, X, d, g, h, std::move(ri), clin, depth + 1, lo, hi);
    arena[ni].left = L; arena[ni].right = R;
    return ni;
  }

  std::mt19937 rng_;  // fit/fit_more 연속 사용 (warm-start 시 subsample 시퀀스 이어짐)

  // raw 점수 누적 (op[i] = init + Σ lr·tree). tree-outer 청크 순회 — 각 트리의
  // 노드 배열이 한 청크 전체 동안 캐시 hot 유지 → 배치 추론 ~1.5x (비트동일).
  void accumulate_raw(const double* X, int n, int d, double* op) {
    double lr = P.learning_rate;
    bool is_mc = (P.objective == 2);
    int K = K_;
    auto block = [&](size_t lo, size_t hi) {
      if (is_mc) {
        bool has_init = (int)init_mc_.size() == K;
        for (size_t i = lo; i < hi; i++) {
          for (int k = 0; k < K; k++) {
            op[i * K + k] = has_init ? init_mc_[k] : 0.0;
          }
        }
        for (auto& tr : trees) {
          for (size_t i = lo; i < hi; i++) {
            int li = leaf_index(tr, X + i * d, dirs_);
            const auto& nd = tr[li];
            if ((int)nd.leaf_weights.size() == K) {
              for (int k = 0; k < K; k++) {
                op[i * K + k] += lr * nd.leaf_weights[k];
              }
            }
          }
        }
      } else {
        for (size_t i = lo; i < hi; i++) op[i] = init_score;
        for (auto& tr : trees) {
          for (size_t i = lo; i < hi; i++) {
            op[i] += lr * predict_one(tr, X + i * d, dirs_);
          }
        }
      }
    };
#ifdef _OPENMP
    if (n > 2000) {
#pragma omp parallel
      {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        size_t lo = (size_t)n * tid / nt, hi = (size_t)n * (tid + 1) / nt;
        block(lo, hi);
      }
      return;
    }
#endif
    block(0, (size_t)n);
  }

  py::array_t<double> predict_raw(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xa) {
    auto Xb = Xa.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    bool is_mc = (P.objective == 2);
    if (is_mc) {
      auto out = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)K_});
      double* op = (double*)out.request().ptr;
      accumulate_raw(X, n, d, op);
      return out;
    } else {
      auto out = py::array_t<double>(n);
      double* op = (double*)out.request().ptr;
      accumulate_raw(X, n, d, op);
      if (P.clip) {
        for (int i = 0; i < n; i++) {
          op[i] = op[i] < y_lo ? y_lo : (op[i] > y_hi ? y_hi : op[i]);
        }
      }
      return out;
    }
  }

  py::array_t<double> predict_proba(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xa) {
    auto Xb = Xa.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    bool is_mc = (P.objective == 2);
    int K = K_;
    if (is_mc) {
      auto out = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)K});
      double* op = (double*)out.request().ptr;
      std::vector<double> raw(n * K);
      accumulate_raw(X, n, d, raw.data());
#pragma omp parallel for if (n * K > 4000)
      for (int i = 0; i < n; i++) {
        double max_val = raw[i * K];
        for (int k = 1; k < K; k++) {
          if (raw[i * K + k] > max_val) max_val = raw[i * K + k];
        }
        double sum_exp = 0.0;
        for (int k = 0; k < K; k++) {
          sum_exp += std::exp(raw[i * K + k] - max_val);
        }
        for (int k = 0; k < K; k++) {
          op[i * K + k] = std::exp(raw[i * K + k] - max_val) / sum_exp;
        }
      }
      return out;
    } else {
      auto out = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)2});
      double* op = (double*)out.request().ptr;
      std::vector<double> raw(n);
      accumulate_raw(X, n, d, raw.data());
#pragma omp parallel for if (n > 2000)
      for (int i = 0; i < n; i++) {
        double p = 1.0 / (1.0 + std::exp(-raw[i]));
        op[i * 2] = 1 - p;
        op[i * 2 + 1] = p;
      }
      return out;
    }
  }

  py::array_t<double> feature_importances() const {
    int d = (int)importances_.size();
    auto out = py::array_t<double>(d);
    double* op = (double*)out.request().ptr;
    double s = 0;
    for (double v : importances_) s += v;
    for (int i = 0; i < d; i++) op[i] = s > 0 ? importances_[i] / s : 0.0;
    return out;
  }

  // 계수 가중 importance: Σ gain·|coef| (정규화). oblique 방향 기여 반영.
  py::array_t<double> coefficient_importances() const {
    int d = (int)coef_imp_.size();
    auto out = py::array_t<double>(d);
    double* op = (double*)out.request().ptr;
    double s = 0;
    for (double v : coef_imp_) s += v;
    for (int i = 0; i < d; i++) op[i] = s > 0 ? coef_imp_[i] / s : 0.0;
    return out;
  }

  // 사선쌍 interaction 행렬 d×d 상삼각: Σ gain·|a|·|b| (정규화). 계산비용 0(누적).
  py::array_t<double> interaction_importances() const {
    auto out = py::array_t<double>({(py::ssize_t)d_, (py::ssize_t)d_});
    double* op = (double*)out.request().ptr;
    double s = 0;
    for (double v : inter_) s += v;
    for (size_t i = 0; i < inter_.size(); i++) op[i] = s > 0 ? inter_[i] / s : 0.0;
    return out;
  }

  // explain: 표본별 가산적 피처 기여 (n, d). 각 트리 기여 lr·w_leaf를 경유한
  // 분할 피처들에 gain·|coef| 비율로 분배 → Σ_i φ_i = 예측 − init_score (SHAP처럼
  // 가산적). "왜 이 예측?"에 답하며 타 모델 SHAP과 직접 비교 가능.
  py::array_t<double> explain(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xa) {
    auto Xb = Xa.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    auto out = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)d});
    double* op = (double*)out.request().ptr;
    std::fill(op, op + (size_t)n * d, 0.0);
    double lr = P.learning_rate;
    int cap = 4 * P.max_depth + 8;  // 경로 최대 분할 피처 수 여유

#pragma omp parallel for if (n > 2000)
    for (int i = 0; i < n; i++) {
      const double* x = X + (size_t)i * d;
      double* phi = op + (size_t)i * d;
      std::vector<int> pf(cap);
      std::vector<double> pr(cap);
      for (auto& tr : trees) {
        int ni = 0, np = 0;
        double tot = 0;
        while (!tr[ni].is_leaf) {
          const Node& nd = tr[ni];
          int ch;
          if (nd.dir_id >= 0) {           // LOB: 라우팅만 (귀속은 Python서 차단)
            ch = lob_child(nd, dirs_, x);
          } else if (nd.type == 1) {
            bool nan = std::isnan(x[nd.fA]);
            ch = nan ? nd.nan_direction : (x[nd.fA] < nd.thr ? 0 : 1);
            if (!nan && np < cap) { pf[np] = nd.fA; pr[np] = nd.gain; tot += nd.gain; np++; }
          } else {
            bool nan = std::isnan(x[nd.fA]) || std::isnan(x[nd.fB]);
            if (nan) {
              ch = nd.nan_direction;
            } else {
              double s = nd.coefA * x[nd.fA] + nd.coefB * x[nd.fB] + nd.bias;
              ch = s < 0 ? 0 : 1;
              double rA = nd.gain * std::fabs(nd.coefA), rB = nd.gain * std::fabs(nd.coefB);
              if (np < cap) { pf[np] = nd.fA; pr[np] = rA; tot += rA; np++; }
              if (np < cap) { pf[np] = nd.fB; pr[np] = rB; tot += rB; np++; }
            }
          }
          ni = ch == 0 ? nd.left : nd.right;
        }
        double c = lr * tr[ni].weight;  // 이 트리의 예측 기여
        if (tot > 0)
          for (int k = 0; k < np; k++) phi[pf[k]] += c * pr[k] / tot;
      }
    }
    return out;
  }

  // ── 직렬화 (predict에 필요한 상태만: init_score, lr, objective, trees) ──
  // Node는 POD라 memcpy 가능. 동일 플랫폼/버전 간 pickle 용도.
  py::bytes serialize() const {
    std::string buf;
    auto wr = [&](const void* p, size_t n) { buf.append((const char*)p, n); };
    wr(&init_score, sizeof(double));
    wr(&P.learning_rate, sizeof(double));
    wr(&P.objective, sizeof(int));
    wr(&P.clip, sizeof(int));
    wr(&y_lo, sizeof(double));
    wr(&y_hi, sizeof(double));
    wr(&K_, sizeof(int));
    int di = (int)importances_.size();
    wr(&di, sizeof(int));
    if (di) wr(importances_.data(), (size_t)di * sizeof(double));
    // 설명용 누적치 (coef importance, interaction d×d, d_)
    wr(&d_, sizeof(int));
    if (di) wr(coef_imp_.data(), (size_t)di * sizeof(double));
    int isz = (int)inter_.size();
    wr(&isz, sizeof(int));
    if (isz) wr(inter_.data(), (size_t)isz * sizeof(double));
    int nt = (int)trees.size();
    wr(&nt, sizeof(int));
    for (auto& tr : trees) {
      int nn = (int)tr.size();
      wr(&nn, sizeof(int));
      for (const auto& nd : tr) {
        wr(&nd.is_leaf, sizeof(bool));
        wr(&nd.weight, sizeof(double));
        wr(&nd.type, sizeof(int));
        wr(&nd.fA, sizeof(int));
        wr(&nd.fB, sizeof(int));
        wr(&nd.thr, sizeof(double));
        wr(&nd.coefA, sizeof(double));
        wr(&nd.coefB, sizeof(double));
        wr(&nd.bias, sizeof(double));
        wr(&nd.left, sizeof(int));
        wr(&nd.right, sizeof(int));
        wr(&nd.nan_direction, sizeof(int));
        wr(&nd.gain, sizeof(double));
        wr(&nd.dir_id, sizeof(int));
        int lw_sz = (int)nd.leaf_weights.size();
        wr(&lw_sz, sizeof(int));
        if (lw_sz) wr(nd.leaf_weights.data(), (size_t)lw_sz * sizeof(double));
      }
    }
    // LOB dense 방향 테이블
    int nd = (int)dirs_.size();
    wr(&nd, sizeof(int));
    for (auto& v : dirs_) {
      int vs = (int)v.size();
      wr(&vs, sizeof(int));
      if (vs) wr(v.data(), (size_t)vs * sizeof(double));
    }
    // 클래스별 log-prior 베이스라인 (구버전 직렬화엔 없음 → deserialize에서 옵셔널)
    int imc = (int)init_mc_.size();
    wr(&imc, sizeof(int));
    if (imc) wr(init_mc_.data(), (size_t)imc * sizeof(double));
    return py::bytes(buf);
  }

  void deserialize(py::bytes b) {
    std::string buf = b;
    const char* p = buf.data();
    size_t off = 0;
    auto rd = [&](void* d, size_t n) {
      std::memcpy(d, p + off, n);
      off += n;
    };
    rd(&init_score, sizeof(double));
    rd(&P.learning_rate, sizeof(double));
    rd(&P.objective, sizeof(int));
    rd(&P.clip, sizeof(int));
    rd(&y_lo, sizeof(double));
    rd(&y_hi, sizeof(double));
    rd(&K_, sizeof(int));
    int di;
    rd(&di, sizeof(int));
    importances_.resize(di);
    if (di) rd(importances_.data(), (size_t)di * sizeof(double));
    rd(&d_, sizeof(int));
    coef_imp_.resize(di);
    if (di) rd(coef_imp_.data(), (size_t)di * sizeof(double));
    int isz;
    rd(&isz, sizeof(int));
    inter_.resize(isz);
    if (isz) rd(inter_.data(), (size_t)isz * sizeof(double));
    int nt;
    rd(&nt, sizeof(int));
    trees.assign(nt, {});
    for (int t = 0; t < nt; t++) {
      int nn;
      rd(&nn, sizeof(int));
      trees[t].resize(nn);
      for (int i = 0; i < nn; i++) {
        auto& nd = trees[t][i];
        rd(&nd.is_leaf, sizeof(bool));
        rd(&nd.weight, sizeof(double));
        rd(&nd.type, sizeof(int));
        rd(&nd.fA, sizeof(int));
        rd(&nd.fB, sizeof(int));
        rd(&nd.thr, sizeof(double));
        rd(&nd.coefA, sizeof(double));
        rd(&nd.coefB, sizeof(double));
        rd(&nd.bias, sizeof(double));
        rd(&nd.left, sizeof(int));
        rd(&nd.right, sizeof(int));
        rd(&nd.nan_direction, sizeof(int));
        rd(&nd.gain, sizeof(double));
        rd(&nd.dir_id, sizeof(int));
        int lw_sz;
        rd(&lw_sz, sizeof(int));
        nd.leaf_weights.resize(lw_sz);
        if (lw_sz) rd(nd.leaf_weights.data(), (size_t)lw_sz * sizeof(double));
      }
    }
    int nd;
    rd(&nd, sizeof(int));
    dirs_.assign(nd, {});
    for (int t = 0; t < nd; t++) {
      int vs;
      rd(&vs, sizeof(int));
      dirs_[t].resize(vs);
      if (vs) rd(dirs_[t].data(), (size_t)vs * sizeof(double));
    }
    // init_mc_ 는 옵셔널 (구버전 호환): 남은 바이트 있으면 읽고, 없으면 0 (= 기존 동작)
    if (off < buf.size()) {
      int imc;
      rd(&imc, sizeof(int));
      init_mc_.resize(imc);
      if (imc) rd(init_mc_.data(), (size_t)imc * sizeof(double));
    } else {
      init_mc_.assign(K_, 0.0);
    }
  }
};

// ── target encoding kernel ──────────────────────────────────────────────────
// Random fold assignment for cross-fitting. stratified=true keeps class
// proportions per fold (group by rounded label, shuffle, block-assign); else
// plain shuffled K-fold. Exact composition is unimportant — only balance.
static py::array_t<int> make_folds(
    py::array_t<double, py::array::c_style | py::array::forcecast> y_a,
    int nf, unsigned seed, bool stratified) {
  auto yb = y_a.request();
  int n = (int)yb.shape[0];
  const double* y = (const double*)yb.ptr;
  auto out = py::array_t<int>(n); int* f = (int*)out.request().ptr;
  std::mt19937 rng(seed);
  if (!stratified) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int r = 0; r < n; r++) f[idx[r]] = (int)((long long)r * nf / n);
  } else {
    std::unordered_map<long long, std::vector<int>> buckets;
    for (int i = 0; i < n; i++) buckets[(long long)std::llround(y[i])].push_back(i);
    for (auto& kv : buckets) {
      auto& v = kv.second;
      std::shuffle(v.begin(), v.end(), rng);
      int sz = (int)v.size();
      for (int r = 0; r < sz; r++) f[v[r]] = (int)((long long)r * nf / sz);
    }
  }
  return out;
}


// Empirical-Bayes (Micci-Barreca "auto") target encoding. Within-level sum of
// squared deviations is computed single-pass as ssd = Q - S^2/C (Q = sum y^2),
// so out-of-fold stats are just global-minus-fold subtractions.
static inline double te_auto(double c, double s, double q, double gmean, double yvar) {
  if (c <= 0) return gmean;
  double mean = s / c;
  double within = (q - s * s / c) / c;       // within-level variance
  double denom = yvar * c + within;
  if (denom <= 0) return gmean;
  double lam = yvar * c / denom;              // shrinkage toward the global mean
  return lam * mean + (1.0 - lam) * gmean;
}

// Cross-fitted encoding for training (leak-resistant via the fold assignment)
// plus the full-data level map for transform() at predict. Returns
// (enc_train[n], full_map[K], global_mean).
static py::tuple te_fit_transform(
    py::array_t<long long, py::array::c_style | py::array::forcecast> codes_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_a,
    py::array_t<int, py::array::c_style | py::array::forcecast> folds_a,
    int K, int nf) {
  auto cb = codes_a.request(), yb = y_a.request(), fb = folds_a.request();
  int n = (int)cb.shape[0];
  const long long* codes = (const long long*)cb.ptr;
  const double* y = (const double*)yb.ptr;
  const int* folds = (const int*)fb.ptr;

  std::vector<double> C((size_t)nf * K, 0), S((size_t)nf * K, 0), Q((size_t)nf * K, 0);
  std::vector<double> Cf(nf, 0), Sf(nf, 0), Qf(nf, 0);
  for (int i = 0; i < n; i++) {
    int f = folds[i]; long long k = codes[i]; double yi = y[i];
    if (k < 0 || k >= K) continue;
    size_t idx = (size_t)f * K + k;
    C[idx] += 1; S[idx] += yi; Q[idx] += yi * yi;
    Cf[f] += 1; Sf[f] += yi; Qf[f] += yi * yi;
  }
  std::vector<double> Ck(K, 0), Sk(K, 0), Qk(K, 0);
  double Ct = 0, St = 0, Qt = 0;
  for (int f = 0; f < nf; f++) {
    Ct += Cf[f]; St += Sf[f]; Qt += Qf[f];
    for (int k = 0; k < K; k++) {
      size_t idx = (size_t)f * K + k; Ck[k] += C[idx]; Sk[k] += S[idx]; Qk[k] += Q[idx];
    }
  }
  std::vector<double> gm(nf), yv(nf);          // per-fold out-of-fold mean / variance
  for (int f = 0; f < nf; f++) {
    double c = Ct - Cf[f], s = St - Sf[f], q = Qt - Qf[f];
    gm[f] = c > 0 ? s / c : 0; yv[f] = c > 0 ? q / c - gm[f] * gm[f] : 0;
  }
  auto enc = py::array_t<double>(n); double* e = (double*)enc.request().ptr;
  for (int i = 0; i < n; i++) {
    int f = folds[i]; long long k = codes[i];
    if (k < 0 || k >= K) { e[i] = gm[f]; continue; }
    size_t idx = (size_t)f * K + k;
    e[i] = te_auto(Ck[k] - C[idx], Sk[k] - S[idx], Qk[k] - Q[idx], gm[f], yv[f]);
  }
  double gmean = Ct > 0 ? St / Ct : 0, yvar = Ct > 0 ? Qt / Ct - gmean * gmean : 0;
  auto full = py::array_t<double>(K); double* fm = (double*)full.request().ptr;
  for (int k = 0; k < K; k++) fm[k] = te_auto(Ck[k], Sk[k], Qk[k], gmean, yvar);
  return py::make_tuple(enc, full, gmean);
}

// Apply a fitted level map; unseen / out-of-range codes fall back to gmean.
static py::array_t<double> te_transform(
    py::array_t<long long, py::array::c_style | py::array::forcecast> codes_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> full_a, double gmean) {
  auto cb = codes_a.request(), mb = full_a.request();
  int n = (int)cb.shape[0], K = (int)mb.shape[0];
  const long long* codes = (const long long*)cb.ptr;
  const double* fm = (const double*)mb.ptr;
  auto out = py::array_t<double>(n); double* o = (double*)out.request().ptr;
  for (int i = 0; i < n; i++) {
    long long k = codes[i];
    o[i] = (k >= 0 && k < K) ? fm[k] : gmean;
  }
  return out;
}

PYBIND11_MODULE(oqboost_core, m) {
  m.def("make_folds", &make_folds, py::arg("y"), py::arg("nf"), py::arg("seed"),
        py::arg("stratified"));
  m.def("te_fit_transform", &te_fit_transform, py::arg("codes"), py::arg("y"),
        py::arg("folds"), py::arg("K"), py::arg("nf"));
  m.def("te_transform", &te_transform, py::arg("codes"), py::arg("full_map"),
        py::arg("gmean"));
  py::class_<Booster>(m, "Booster")
      .def(py::init<int, double, int, int, double, int, int, double, double,
                    unsigned, int, int, int, double, int, std::vector<int>,
                    std::vector<int>, int>(),
           py::arg("n_estimators") = 60, py::arg("learning_rate") = 0.12,
           py::arg("max_depth") = 4, py::arg("max_bins") = 64,
           py::arg("reg_lambda") = 1.0, py::arg("min_samples") = 10,
           py::arg("n_screen") = -1, py::arg("subsample") = 1.0,
           py::arg("colsample") = 1.0, py::arg("seed") = 42,
           py::arg("objective") = 0, py::arg("fast_dir") = 1,
           py::arg("loss") = 0, py::arg("alpha") = 0.9, py::arg("clip") = 0,
           py::arg("monotone") = std::vector<int>{},
           py::arg("categorical") = std::vector<int>{},
           py::arg("max_lineage") = 0)
      .def("fit", &Booster::fit, py::arg("X"), py::arg("y"),
           py::arg("sample_weight") = py::array_t<double>(0),
           py::arg("X_val") = py::array_t<double>(0),
           py::arg("y_val") = py::array_t<double>(0),
           py::arg("es_patience") = 0, py::arg("es_tol") = 1e-4)
      .def("fit_more", &Booster::fit_more, py::arg("X"), py::arg("y"),
           py::arg("extra"), py::arg("sample_weight") = py::array_t<double>(0))
      .def("n_trees", &Booster::n_trees)
      .def("best_iteration", [](const Booster& b) { return b.best_iteration_; })
      .def("predict_raw", &Booster::predict_raw)
      .def("predict_proba", &Booster::predict_proba)
      .def("feature_importances", &Booster::feature_importances)
      .def("coefficient_importances", &Booster::coefficient_importances)
      .def("interaction_importances", &Booster::interaction_importances)
      .def("explain", &Booster::explain)
      .def("serialize", &Booster::serialize)
      .def("deserialize", &Booster::deserialize);
}