// oqboost_core.cpp — OQBoost 2.0 core (histogram-binned 2D-oblique GBDT)
// 2D-oblique Newton-boosted GBDT. 전역 사전 binning(히스토그램 트릭)으로
// 노드별 정렬 제거. 범주 서브시스템 없음(정수코드=연속). pybind11.
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

static Bins precompute_bins(const double* X, int n, int d, int max_bins) {
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
    int actual_max_bins = fnan ? std::max(2, max_bins - 1) : max_bins;
    for (int b = 1; b < actual_max_bins; b++)
      e.push_back(percentile_sorted(cs, 100.0 * b / actual_max_bins));
    e = unique_sorted(e);

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
};
struct Params {
  int n_estimators = 60, max_depth = 4, max_bins = 64, min_samples = 10;
  double learning_rate = 0.12, reg_lambda = 1.0;
  int n_screen = -1;
  double subsample = 1.0, colsample = 1.0;
  unsigned seed = 42;
  int objective = 0;
  int fast_dir = 0;  // 1=H-가중 gradient 회귀 방향(그리드/BHC 생략), 0=BHC seed
  // 회귀 손실: 0=squared(L2), 1=huber(robust), 2=quantile(pinball).
  int loss = 0;
  double alpha = 0.9;  // huber: |residual| delta 분위 / quantile: 목표 분위
  int clip = 0;        // 1=예측을 train 타깃 [min,max]로 clamp (외삽 폭주 방지)
  // 피처별 단조 제약: -1=감소, 0=무제약, +1=증가. 비면 제약 없음(fast path).
  std::vector<int> monotone;
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
};

// ─── SIS 스크리닝 (캐시 프렌들리 Row-major 순회로 대폭 최적화)
// ────────────────
static std::vector<int> screen(const double* X, int d,
                               const std::vector<int>& idx,
                               const std::vector<double>& g, int m) {
  if (m < 0 || m >= d) {
    std::vector<int> a(d);
    std::iota(a.begin(), a.end(), 0);
    return a;
  }
  // m=0과 m=1을 동일하게 취급: 둘 다 "피처 1개만 통과"이므로 eval_2d가
  // pair를 만들 후보가 없어 자연히 단일 축(축 정렬) 분할로 회귀한다.
  // 이전에는 m=0이 fs.resize(0)으로 빈 피처 목록을 반환해 모든 노드가
  // 즉시 leaf가 되고 모델이 완전히 무력화(상수 예측)되는 버그가 있었음.
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

    // Pass 1: 샘플(행)을 외곽 루프에 두어 메모리를 연속적으로 읽음 (캐시
    // 최적화)
    for (int i = 0; i < n; i++) {
      size_t row_idx = (size_t)idx[i] * d;
      for (int f = 0; f < d; f++) {
        double v = X[row_idx + f];
        if (!std::isnan(v)) {
          xm[f] += v;
          valid_count[f]++;
        }
      }
    }

    for (int f = 0; f < d; f++) {
      if (valid_count[f] >= 2) xm[f] /= valid_count[f];
    }

    std::vector<double> xv(d, 0.0);
    std::vector<double> cov(d, 0.0);

    // Pass 2: 분산 및 공분산 계산도 동일하게 캐시 친화적 구조로 변경
    for (int i = 0; i < n; i++) {
      size_t row_idx = (size_t)idx[i] * d;
      double g_diff = g[idx[i]] - gm;
      for (int f = 0; f < d; f++) {
        if (valid_count[f] < 2) continue;
        double v = X[row_idx + f];
        if (!std::isnan(v)) {
          double xt = v - xm[f];
          xv[f] += xt * xt;
          cov[f] += xt * g_diff;
        }
      }
    }

    for (int f = 0; f < d; f++) {
      if (valid_count[f] < 2) {
        score[f] = 0;
        continue;
      }
      double xs = std::sqrt(xv[f] / valid_count[f]);
      score[f] =
          (xs > 1e-12) ? std::fabs(cov[f] / valid_count[f] / (xs * gstd)) : 0;
    }
  }
  std::vector<int> fs(d);
  std::iota(fs.begin(), fs.end(), 0);
  std::partial_sort(fs.begin(), fs.begin() + m, fs.end(),
                    [&](int a, int b) { return score[a] > score[b]; });
  fs.resize(m);
  std::sort(fs.begin(), fs.end());
  return fs;
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
static bool refine_threshold(const std::vector<double>& proj, const double* gn,
                             const double* hn, double lam, double Gp, double Hp,
                             double& outT, double& outGain, bool has_nan,
                             Workspace& ws) {
  int n = (int)proj.size();
  double mn = 0.0, mx = 0.0;
  bool first = true;

  if (!has_nan) {
    if (n == 0) return false;
    mn = proj[0];
    mx = proj[0];
    for (int i = 1; i < n; i++) {
      double v = proj[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    first = false;
  } else {
    for (double v : proj) {
      if (!std::isnan(v)) {
        if (first) {
          mn = v;
          mx = v;
          first = false;
        } else {
          mn = std::min(mn, v);
          mx = std::max(mx, v);
        }
      }
    }
  }
  if (first || (mx - mn < 1e-12)) return false;

  const int B = g_hist_bins;
  double w = (mx - mn) / B;
  ws.histG.assign(B, 0.0);
  ws.histH.assign(B, 0.0);
  std::vector<double>& Gb = ws.histG;
  std::vector<double>& Hb = ws.histH;
  double G_nan = 0, H_nan = 0;

  if (!has_nan) {
    // NaN이 없는 경우 분기문 없는 순수 SIMD 고속 루프 가능
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

// ─── 노드 피처 캐시 ─────────────────────────────────────────────────────────
struct FCache {
  int f;
  std::vector<double> col;
  std::vector<u16> bin;
};

static std::vector<FCache> build_caches(const double* X, int d,
                                        const std::vector<u16>& binidx,
                                        const std::vector<int>& idx,
                                        const std::vector<int>& feats) {
  std::vector<FCache> C(feats.size());
  size_t n_samples = idx.size();
  for (size_t fi = 0; fi < feats.size(); fi++) {
    FCache& c = C[fi];
    c.f = feats[fi];
    c.col.resize(n_samples);
    c.bin.resize(n_samples);
  }

  // 이 함수 역시 샘플(행)을 외곽 루프에 두어 캐시 적중률 극대화
  for (size_t i = 0; i < n_samples; i++) {
    size_t row_offset = (size_t)idx[i] * d;
    for (size_t fi = 0; fi < feats.size(); fi++) {
      int f = C[fi].f;
      C[fi].col[i] = X[row_offset + f];
      C[fi].bin[i] = binidx[row_offset + f];
    }
  }
  return C;
}

// ─── 1D 분할 ─────────────────────────────────────────────────────────────────
static Split eval_1d(const std::vector<FCache>& C,
                     const std::vector<std::vector<double>>& centers,
                     const std::vector<double>& gn,
                     const std::vector<double>& hn, double Gp, double Hp,
                     const Params& P) {
  Split best;
  double base = gain_term(Gp, Hp, P.reg_lambda);
  int nloc = (int)gn.size();

  for (const FCache& c : C) {
    const std::vector<double>& ctr = centers[c.f];
    int k = (int)ctr.size();
    if (k < 2) continue;

    std::vector<double> Ga(k, 0), Ha(k, 0);
    for (int i = 0; i < nloc; i++) {
      int b = c.bin[i];
      Ga[b] += gn[i];
      Ha[b] += hn[i];
    }

    std::vector<int> occ;
    for (int a = 0; a < k; a++) {
      if (Ha[a] > 0.0) occ.push_back(a);  // h>0 ⟺ 점유
    }
    if ((int)occ.size() < 2) continue;

    double GL = 0, HL = 0;
    for (int ki = 0; ki + 1 < (int)occ.size(); ki++) {
      GL += Ga[occ[ki]];
      HL += Ha[occ[ki]];
      if (HL <= 1e-12 || (Hp - HL) <= 1e-12) continue;
      double gain_val = gain_term(GL, HL, P.reg_lambda) +
                        gain_term(Gp - GL, Hp - HL, P.reg_lambda) - base;
      if (gain_val > best.gain) {
        best.gain = gain_val;
        best.type = 1;
        best.fA = c.f;
        best.thr = (ctr[occ[ki]] + ctr[occ[ki + 1]]) / 2.0;
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
  int nloc = (int)gn.size(), K = kA * kB;
  double coefA, coefB;

  if (P.fast_dir) {
    // H-가중 gradient 직접 회귀: t=-g/h를 두 피처에 가중 LSQ → 방향.
    // 그리드 scatter·점유수집·정렬·BHC·LSQ 전부 생략, 9-스칼라 1패스 + 2×2.
    double Sh = 0, Sa = 0, Sb = 0, Saa = 0, Sab = 0, Sbb = 0, Sat = 0, Sbt = 0,
           St = 0;
    for (int i = 0; i < nloc; i++) {
      double xa = cA_.col[i], xb = cB_.col[i];
      if (has_nan && (std::isnan(xa) || std::isnan(xb))) continue;
      double gi = gn[i], hi = hn[i];
      Sh += hi;
      Sa += hi * xa;
      Sb += hi * xb;
      Saa += hi * xa * xa;
      Sab += hi * xa * xb;
      Sbb += hi * xb * xb;
      Sat += -gi * xa;
      Sbt += -gi * xb;
      St += -gi;  // h·t=-g
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
  } else {
    // h>0 항상참 → Hc>0 ⟺ 점유. cnt 배열 제거(샘플당 scatter write 3→2).
    ws.Gc.assign(K, 0);
    ws.Hc.assign(K, 0);
    for (int i = 0; i < nloc; i++) {
      int c = cA_.bin[i] * kB + cB_.bin[i];
      ws.Gc[c] += gn[i];
      ws.Hc[c] += hn[i];
    }
    ws.oa.clear();
    ws.ob.clear();
    ws.Gs.clear();
    ws.Hs.clear();
    for (int a = 0; a < kA; a++) {
      for (int b = 0; b < kB; b++) {
        int c = a * kB + b;
        if (ws.Hc[c] > 0.0) {
          ws.oa.push_back(a);
          ws.ob.push_back(b);
          ws.Gs.push_back(ws.Gc[c]);
          ws.Hs.push_back(ws.Hc[c]);
        }
      }
    }
    int S = (int)ws.oa.size();
    if (S < 2) return s;
    ws.so.resize(S);
    std::iota(ws.so.begin(), ws.so.end(), 0);
    std::sort(ws.so.begin(), ws.so.end(), [&](int a, int b) {
      return -ws.Gs[a] / (ws.Hs[a] + P.reg_lambda) <
             -ws.Gs[b] / (ws.Hs[b] + P.reg_lambda);
    });
    double base = gain_term(Gp, Hp, P.reg_lambda), GL = 0, HL = 0, bg = 0;
    int bk = -1;
    for (int ki = 0; ki + 1 < S; ki++) {
      GL += ws.Gs[ws.so[ki]];
      HL += ws.Hs[ws.so[ki]];
      double gv = gain_term(GL, HL, P.reg_lambda) +
                  gain_term(Gp - GL, Hp - HL, P.reg_lambda) - base;
      if (gv > bg) {
        bg = gv;
        bk = ki;
      }
    }
    if (bk < 0) return s;
    ws.lab.assign(S, 1);
    for (int j = 0; j <= bk; j++) ws.lab[ws.so[j]] = 0;
    ws.cA.resize(S);
    ws.cB.resize(S);
    for (int t = 0; t < S; t++) {
      ws.cA[t] = ctrA[ws.oa[t]];
      ws.cB[t] = ctrB[ws.ob[t]];
    }
    if (!lsq_separator(ws.cA, ws.cB, ws.lab, ws.Hs, coefA, coefB)) return s;
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
  if (!has_nan) {
// NaN 없으면 벡터화 힌트 (MSVC /openmp는 omp simd 미지원 → GCC/Clang만)
#if defined(__GNUC__) || defined(__clang__)
#pragma omp simd
#endif
    for (int i = 0; i < nloc; i++) {
      ws.proj[i] = coefA * cA_.col[i] + coefB * cB_.col[i];
    }
  } else {
    for (int i = 0; i < nloc; i++) {
      if (std::isnan(cA_.col[i]) || std::isnan(cB_.col[i])) {
        ws.proj[i] = std::numeric_limits<double>::quiet_NaN();
      } else {
        ws.proj[i] = coefA * cA_.col[i] + coefB * cB_.col[i];
      }
    }
  }
  double t, gn2;
  if (!refine_threshold(ws.proj, gn.data(), hn.data(), P.reg_lambda, Gp, Hp, t,
                        gn2, has_nan, ws))
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
                     const Params& P) {
  int nf = (int)C.size();
  std::vector<std::pair<int, int>> pr;
  pr.reserve(nf * (nf - 1) / 2);
  for (int a = 0; a < nf; a++)
    for (int b = a + 1; b < nf; b++) pr.emplace_back(a, b);
  int np = (int)pr.size();
  int nloc = (int)gn.size();
  std::vector<Split> res(np);

  int max_threads = 1;
#ifdef _OPENMP
  max_threads = omp_get_max_threads();
#endif
  // 작업량(쌍수×표본수)이 작으면 fork-join 오버헤드 > 이득 → serial 폴백.
  // 깊은 노드(소표본)·작은 데이터서 스레드 생성 비용을 제거(small-data 회귀 방지).
  bool par = max_threads > 1 && (long)np * nloc > 30000;
  // 스레드별 개별 Workspace (thread_local 병목 회피). serial이면 1개만 할당.
  std::vector<Workspace> wss(par ? max_threads : 1);

#pragma omp parallel for schedule(dynamic, 4) if (par)
  for (int p = 0; p < np; p++) {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    int fA = C[pr[p].first].f;
    int fB = C[pr[p].second].f;
    bool pair_has_nan = has_nan[fA] || has_nan[fB];
    res[p] = eval_pair(C[pr[p].first], C[pr[p].second], centers[fA],
                       centers[fB], gn, hn, Gp, Hp, P, wss[tid], pair_has_nan);
  }
  Split best;
  for (const Split& s : res)
    if (s.gain > best.gain) best = s;
  return best;
}

// ─── 재귀 빌드 ───────────────────────────────────────────────────────────────
static int build(std::vector<Node>& arena, const double* X, int d,
                 const std::vector<u16>& binidx,
                 const std::vector<std::vector<double>>& centers,
                 const std::vector<bool>& has_nan, const std::vector<double>& g,
                 const std::vector<double>& h, std::vector<int> idx, int depth,
                 const Params& P, std::mt19937& rng, std::vector<double>& imp,
                 double lo, double hi) {
  double Gp = 0, Hp = 0;
  for (int i : idx) {
    Gp += g[i];
    Hp += h[i];
  }
  int ni = (int)arena.size();
  arena.push_back(Node());
  double w = -Gp / (Hp + P.reg_lambda);
  arena[ni].weight = w < lo ? lo : (w > hi ? hi : w);  // 단조 경계로 clamp
  if (depth >= P.max_depth || (int)idx.size() < P.min_samples) return ni;

  auto feats = screen(X, d, idx, g, P.n_screen);
  if (P.colsample < 1.0 && (int)feats.size() > 2) {
    int keep = std::max(2, (int)std::ceil(P.colsample * feats.size()));
    std::shuffle(feats.begin(), feats.end(), rng);
    feats.resize(keep);
    std::sort(feats.begin(), feats.end());
  }
  auto C = build_caches(X, d, binidx, idx, feats);
  std::vector<double> gn(idx.size()), hn(idx.size());
  for (size_t i = 0; i < idx.size(); i++) {
    gn[i] = g[idx[i]];
    hn[i] = h[idx[i]];
  }
  Split s1 = eval_1d(C, centers, gn, hn, Gp, Hp, P);
  Split s2 = eval_2d(C, centers, has_nan, gn, hn, Gp, Hp, P);
  Split bs = (s2.gain >= s1.gain) ? s2 : s1;
  if (bs.gain <= 1e-6 || bs.type == 0) return ni;

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
                depth + 1, P, rng, imp, lo_l, hi_l);
  int R = build(arena, X, d, binidx, centers, has_nan, g, h, std::move(ri),
                depth + 1, P, rng, imp, lo_r, hi_r);
  arena[ni].left = L;
  arena[ni].right = R;
  return ni;
}

static inline double predict_one(const std::vector<Node>& A, const double* x) {
  int ni = 0;
  while (true) {
    const Node& nd = A[ni];
    if (nd.is_leaf) return nd.weight;
    int ch;
    if (nd.type == 1) {
      if (std::isnan(x[nd.fA]))
        ch = nd.nan_direction;
      else
        ch = (x[nd.fA] < nd.thr) ? 0 : 1;
    } else {
      if (std::isnan(x[nd.fA]) || std::isnan(x[nd.fB])) {
        ch = nd.nan_direction;
      } else {
        double s = nd.coefA * x[nd.fA] + nd.coefB * x[nd.fB] + nd.bias;
        ch = (s < 0) ? 0 : 1;
      }
    }
    ni = (ch == 0) ? nd.left : nd.right;
  }
}

// 표본이 도달하는 leaf의 arena 인덱스 (leaf-value line-search용).
static inline int leaf_index(const std::vector<Node>& A, const double* x) {
  int ni = 0;
  while (true) {
    const Node& nd = A[ni];
    if (nd.is_leaf) return ni;
    int ch;
    if (nd.type == 1) {
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
  std::vector<double> importances_;  // 피처별 누적 gain
  Booster(int n_estimators, double learning_rate, int max_depth, int max_bins,
          double reg_lambda, int min_samples, int n_screen, double subsample,
          double colsample, unsigned seed, int objective, int fast_dir,
          int loss, double alpha, int clip, std::vector<int> monotone) {
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
  }

  void fit(py::array_t<double, py::array::c_style | py::array::forcecast> Xa,
           py::array_t<double, py::array::c_style | py::array::forcecast> ya) {
    auto Xb = Xa.request();
    auto yb = ya.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    const double* y = (const double*)yb.ptr;
    // 단일 소스로 통일: max_bins가 전역 사전 binning과 2D threshold 스캔
    // 해상도를 동시에 결정. fit() 호출당 한 번만 쓰고 이후 읽기 전용.
    g_hist_bins = std::max(2, P.max_bins);
    Bins B = precompute_bins(X, n, d, P.max_bins);

    double ybar = 0;
    for (int i = 0; i < n; i++) ybar += y[i];
    ybar /= n;
    y_lo = y_hi = (n ? y[0] : 0.0);
    for (int i = 0; i < n; i++) { y_lo = std::min(y_lo, y[i]); y_hi = std::max(y_hi, y[i]); }
    if (P.objective == 0) {
      double y2 = std::min(std::max(ybar, 1e-6), 1 - 1e-6);
      init_score = std::log(y2 / (1 - y2));
    } else if (P.loss == 0) {
      init_score = ybar;  // squared → mean
    } else {
      // huber/quantile → median (이상치에 robust한 시작점)
      std::vector<double> ys(y, y + n);
      size_t mid = ys.size() / 2;
      std::nth_element(ys.begin(), ys.begin() + mid, ys.end());
      init_score = ys[mid];
    }
    std::vector<double> raw(n, init_score);
    trees.clear();
    trees.reserve(P.n_estimators);
    importances_.assign(d, 0.0);
    rng_.seed(P.seed);
    boost_rounds(X, y, n, d, B, raw, P.n_estimators);
  }

  // warm-start: 기존 트리 위에 extra 라운드 추가. raw 상태는 저장된 트리로
  // 재구성하므로 별도 상태 보존 불필요. init_score·importances·rng_ 이어서 사용.
  void fit_more(py::array_t<double, py::array::c_style | py::array::forcecast> Xa,
                py::array_t<double, py::array::c_style | py::array::forcecast> ya,
                int extra) {
    auto Xb = Xa.request();
    auto yb = ya.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    const double* y = (const double*)yb.ptr;
    if (extra <= 0 || trees.empty()) return;
    g_hist_bins = std::max(2, P.max_bins);
    Bins B = precompute_bins(X, n, d, P.max_bins);
    // 기존 트리로 raw 재구성 (predict_raw와 동일, clip 제외)
    std::vector<double> raw(n, init_score);
#pragma omp parallel for if (n > 2000)
    for (int i = 0; i < n; i++) {
      const double* x = X + (size_t)i * d;
      for (auto& tr : trees) raw[i] += P.learning_rate * predict_one(tr, x);
    }
    if ((int)importances_.size() != d) importances_.assign(d, 0.0);
    trees.reserve(trees.size() + extra);
    boost_rounds(X, y, n, d, B, raw, extra);
  }

  int n_trees() const { return (int)trees.size(); }

  // 부스팅 라운드 실행기 (fit / fit_more 공유). rng_·trees·importances_·raw 갱신.
  void boost_rounds(const double* X, const double* y, int n, int d,
                    const Bins& B, std::vector<double>& raw, int rounds) {
    std::vector<double> g(n), h(n), absr;
    if (P.objective == 1 && P.loss == 1) absr.resize(n);  // huber delta 계산용
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    int n_sub = std::max(1, (int)(P.subsample * n));

    for (int t = 0; t < rounds; t++) {
      if (P.objective == 0) {
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

      std::vector<int> rows;
      if (P.subsample < 1.0) {
        std::shuffle(all.begin(), all.end(), rng_);
        rows.assign(all.begin(), all.begin() + n_sub);
      } else
        rows = all;
      std::vector<Node> arena;
      arena.reserve(256);
      build(arena, X, d, B.idx, B.centers, B.has_nan, g, h, rows, 0, P, rng_,
            importances_, -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity());

      // quantile: gradient는 부호뿐(±α)이라 -G/H leaf 값이 무의미 → 트리 구조는
      // gradient로 만들되 leaf 값을 멤버 residual의 alpha-분위로 line-search.
      if (P.objective == 1 && P.loss == 2) {
        std::vector<std::vector<double>> res(arena.size());
        for (int i : rows)
          res[leaf_index(arena, X + (size_t)i * d)].push_back(y[i] - raw[i]);
        for (size_t li = 0; li < arena.size(); li++)
          if (arena[li].is_leaf && !res[li].empty())
            arena[li].weight = quantile_inplace(res[li], P.alpha);
      }

#pragma omp parallel for if (n > 2000)
      for (int i = 0; i < n; i++)
        raw[i] += P.learning_rate * predict_one(arena, X + (size_t)i * d);
      trees.push_back(std::move(arena));
    }
  }

  std::mt19937 rng_;  // fit/fit_more 연속 사용 (warm-start 시 subsample 시퀀스 이어짐)

  py::array_t<double> predict_raw(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xa) {
    auto Xb = Xa.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    auto out = py::array_t<double>(n);
    double* op = (double*)out.request().ptr;

#pragma omp parallel for if (n > 2000)
    for (int i = 0; i < n; i++) {
      double r = init_score;
      const double* x = X + (size_t)i * d;
      for (auto& tr : trees) r += P.learning_rate * predict_one(tr, x);
      if (P.clip) r = r < y_lo ? y_lo : (r > y_hi ? y_hi : r);
      op[i] = r;
    }
    return out;
  }

  py::array_t<double> predict_proba(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xa) {
    auto Xb = Xa.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;

    auto out = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)2});
    double* op = (double*)out.request().ptr;

#pragma omp parallel for if (n > 2000)
    for (int i = 0; i < n; i++) {
      double r = init_score;
      const double* x = X + (size_t)i * d;
      for (auto& tr : trees) r += P.learning_rate * predict_one(tr, x);
      double p = 1.0 / (1.0 + std::exp(-r));
      op[i * 2] = 1 - p;
      op[i * 2 + 1] = p;
    }
    return out;
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
    int di = (int)importances_.size();
    wr(&di, sizeof(int));
    if (di) wr(importances_.data(), (size_t)di * sizeof(double));
    int nt = (int)trees.size();
    wr(&nt, sizeof(int));
    for (auto& tr : trees) {
      int nn = (int)tr.size();
      wr(&nn, sizeof(int));
      if (nn) wr(tr.data(), (size_t)nn * sizeof(Node));
    }
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
    int di;
    rd(&di, sizeof(int));
    importances_.resize(di);
    if (di) rd(importances_.data(), (size_t)di * sizeof(double));
    int nt;
    rd(&nt, sizeof(int));
    trees.assign(nt, {});
    for (int t = 0; t < nt; t++) {
      int nn;
      rd(&nn, sizeof(int));
      trees[t].resize(nn);
      if (nn) rd(trees[t].data(), (size_t)nn * sizeof(Node));
    }
  }
};

PYBIND11_MODULE(oqboost_core, m) {
  py::class_<Booster>(m, "Booster")
      .def(py::init<int, double, int, int, double, int, int, double, double,
                    unsigned, int, int, int, double, int, std::vector<int>>(),
           py::arg("n_estimators") = 60, py::arg("learning_rate") = 0.12,
           py::arg("max_depth") = 4, py::arg("max_bins") = 64,
           py::arg("reg_lambda") = 1.0, py::arg("min_samples") = 10,
           py::arg("n_screen") = -1, py::arg("subsample") = 1.0,
           py::arg("colsample") = 1.0, py::arg("seed") = 42,
           py::arg("objective") = 0, py::arg("fast_dir") = 0,
           py::arg("loss") = 0, py::arg("alpha") = 0.9, py::arg("clip") = 0,
           py::arg("monotone") = std::vector<int>{})
      .def("fit", &Booster::fit)
      .def("fit_more", &Booster::fit_more)
      .def("n_trees", &Booster::n_trees)
      .def("predict_raw", &Booster::predict_raw)
      .def("predict_proba", &Booster::predict_proba)
      .def("feature_importances", &Booster::feature_importances)
      .def("serialize", &Booster::serialize)
      .def("deserialize", &Booster::deserialize);
}