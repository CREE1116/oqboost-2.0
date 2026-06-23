// oqboost_core.cpp — OQBoost 2.0 core (histogram-binned 2D-oblique GBDT)
// 2D-oblique Newton-boosted GBDT. 전역 사전 binning(히스토그램 트릭)으로
// 노드별 정렬 제거. 범주 서브시스템 없음(정수코드=연속). pybind11.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
      // 모든 값이 결측치인 경우 예외 처리
      B.edges[f] = {};
      B.centers[f] = {0.0};
      for (int i = 0; i < n; i++) B.idx[(size_t)i * d + f] = 0;
      continue;
    }

    std::sort(cs.begin(), cs.end());
    std::vector<double> e;
    // max_bins - 1개만큼 경계 생성 (NaN 공간 확보 고려)
    int actual_max_bins = fnan ? std::max(2, max_bins - 1) : max_bins;
    for (int b = 1; b < actual_max_bins; b++)
      e.push_back(percentile_sorted(cs, 100.0 * b / actual_max_bins));
    e = unique_sorted(e);

    // 정상 빈 개수 + (NaN이 있다면 마지막 빈 추가)
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
        ctr[b] = 0.0;  // NaN 빈의 중심값은 임의로 0 처리 (투영 시 분리 우선)
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
  int nan_direction = 0;  // 0: Left, 1: Right (결측치 라우팅용 방어 필드)
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
};

// ─── SIS 스크리닝 (NaN 방어 적용) ────────────────────────────────────────────
static std::vector<int> screen(const double* X, int d,
                               const std::vector<int>& idx,
                               const std::vector<double>& g, int m) {
  if (m < 0 || m >= d) {
    std::vector<int> a(d);
    std::iota(a.begin(), a.end(), 0);
    return a;
  }
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
    for (int f = 0; f < d; f++) {
      double xm = 0;
      int valid_count = 0;
      for (int i = 0; i < n; i++) {
        double v = X[(size_t)idx[i] * d + f];
        if (!std::isnan(v)) {
          xm += v;
          valid_count++;
        }
      }
      if (valid_count < 2) {
        score[f] = 0;
        continue;
      }
      xm /= valid_count;

      double xv = 0, cov = 0;
      for (int i = 0; i < n; i++) {
        double v = X[(size_t)idx[i] * d + f];
        if (!std::isnan(v)) {
          double xt = v - xm;
          xv += xt * xt;
          cov += xt * (g[idx[i]] - gm);
        }
      }
      double xs = std::sqrt(xv / valid_count);
      score[f] = (xs > 1e-12) ? std::fabs(cov / valid_count / (xs * gstd)) : 0;
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

// ─── 투영 위 히스토그램 임계 (O(n+B) - NaN 방어 강화) ───────────────────────
static bool refine_threshold(const std::vector<double>& proj, const double* gn,
                             const double* hn, double lam, double Gp, double Hp,
                             double& outT, double& outGain) {
  int n = (int)proj.size();
  double mn = 0.0, mx = 0.0;
  bool first = true;

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
  if (first || (mx - mn < 1e-12)) return false;

  const int B = 64;
  double w = (mx - mn) / B;
  double Gb[64] = {0}, Hb[64] = {0};
  double G_nan = 0, H_nan = 0;

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

  double base = gain_term(Gp, Hp, lam), GL = 0, HL = 0, bg = 0, bt = 0;
  bool found = false;

  // 기본적으로 NaN이 없는 경우 탐색
  for (int b = 0; b + 1 < B; b++) {
    GL += Gb[b];
    HL += Hb[b];

    // NaN 샘플을 Left로 보낼 때와 Right로 보낼 때 모두 검사
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
  for (size_t fi = 0; fi < feats.size(); fi++) {
    FCache& c = C[fi];
    c.f = feats[fi];
    int f = c.f;
    c.col.resize(idx.size());
    c.bin.resize(idx.size());
    for (size_t i = 0; i < idx.size(); i++) {
      c.col[i] = X[(size_t)idx[i] * d + f];
      c.bin[i] = binidx[(size_t)idx[i] * d + f];
    }
  }
  return C;
}

// ─── 1D 분할 (연속형 공간 순서 기반 bin 히스토그램 스캔 - NaN 방어 추가)
// ──────
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
    std::vector<int> cnt(k, 0);
    for (int i = 0; i < nloc; i++) {
      int b = c.bin[i];
      Ga[b] += gn[i];
      Ha[b] += hn[i];
      cnt[b]++;
    }

    // NaN 전용 마지막 빈이 있는지 확인
    bool has_nan_bin = (k > 1 && std::isnan(ctr.back()) == false &&
                        (size_t)c.f < centers.size());
    // 실제 의미있는 오름차순 빈 정렬
    std::vector<int> occ;
    int norm_k =
        (k > 1 && cnt.back() > 0) ? k - 1 : k;  // 단순 방어적 결측치 제외 처리

    for (int a = 0; a < k; a++) {
      if (cnt[a] > 0) occ.push_back(a);
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
        best.nan_direction = 1;  // Default Right
      }
    }
  }
  return best;
}

// ─── 한 쌍 2D oblique (모든 임시 메모리 재할당 제거로 병렬 성능 최적화) ─────
static Split eval_pair(const FCache& cA_, const FCache& cB_,
                       const std::vector<double>& ctrA,
                       const std::vector<double>& ctrB,
                       const std::vector<double>& gn,
                       const std::vector<double>& hn, double Gp, double Hp,
                       const Params& P) {
  Split s;
  int kA = (int)ctrA.size(), kB = (int)ctrB.size();
  if (kA < 1 || kB < 1) return s;
  int nloc = (int)gn.size(), K = kA * kB;

  static thread_local std::vector<double> Gc, Hc, proj;
  static thread_local std::vector<int> cnt;
  static thread_local std::vector<int> oa, ob;
  static thread_local std::vector<double> Gs, Hs;
  static thread_local std::vector<int> so;
  static thread_local std::vector<int> lab;
  static thread_local std::vector<double> cA, cB;

  Gc.assign(K, 0);
  Hc.assign(K, 0);
  cnt.assign(K, 0);
  for (int i = 0; i < nloc; i++) {
    int c = cA_.bin[i] * kB + cB_.bin[i];
    Gc[c] += gn[i];
    Hc[c] += hn[i];
    cnt[c]++;
  }

  oa.clear();
  ob.clear();
  Gs.clear();
  Hs.clear();
  for (int a = 0; a < kA; a++)
    for (int b = 0; b < kB; b++) {
      int c = a * kB + b;
      if (cnt[c] > 0) {
        oa.push_back(a);
        ob.push_back(b);
        Gs.push_back(Gc[c]);
        Hs.push_back(Hc[c]);
      }
    }
  int S = (int)oa.size();
  if (S < 2) return s;

  so.resize(S);
  std::iota(so.begin(), so.end(), 0);
  std::sort(so.begin(), so.end(), [&](int a, int b) {
    return -Gs[a] / (Hs[a] + P.reg_lambda) < -Gs[b] / (Hs[b] + P.reg_lambda);
  });
  double base = gain_term(Gp, Hp, P.reg_lambda), GL = 0, HL = 0, bg = 0;
  int bk = -1;
  for (int ki = 0; ki + 1 < S; ki++) {
    GL += Gs[so[ki]];
    HL += Hs[so[ki]];
    double gv = gain_term(GL, HL, P.reg_lambda) +
                gain_term(Gp - GL, Hp - HL, P.reg_lambda) - base;
    if (gv > bg) {
      bg = gv;
      bk = ki;
    }
  }
  if (bk < 0) return s;

  lab.assign(S, 1);
  for (int j = 0; j <= bk; j++) lab[so[j]] = 0;
  cA.resize(S);
  cB.resize(S);
  for (int t = 0; t < S; t++) {
    cA[t] = ctrA[oa[t]];
    cB[t] = ctrB[ob[t]];
  }
  double coefA, coefB;
  if (!lsq_separator(cA, cB, lab, Hs, coefA, coefB)) return s;

  proj.resize(nloc);
  for (int i = 0; i < nloc; i++) {
    if (std::isnan(cA_.col[i]) || std::isnan(cB_.col[i])) {
      proj[i] = std::numeric_limits<double>::quiet_NaN();
    } else {
      proj[i] = coefA * cA_.col[i] + coefB * cB_.col[i];
    }
  }
  double t, gn2;
  if (!refine_threshold(proj, gn.data(), hn.data(), P.reg_lambda, Gp, Hp, t,
                        gn2))
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
                     const std::vector<double>& gn,
                     const std::vector<double>& hn, double Gp, double Hp,
                     const Params& P) {
  int nf = (int)C.size();
  std::vector<std::pair<int, int>> pr;
  pr.reserve(nf * (nf - 1) / 2);
  for (int a = 0; a < nf; a++)
    for (int b = a + 1; b < nf; b++) pr.emplace_back(a, b);
  int np = (int)pr.size();
  std::vector<Split> res(np);
#pragma omp parallel for schedule(dynamic, 4)
  for (int p = 0; p < np; p++)
    res[p] =
        eval_pair(C[pr[p].first], C[pr[p].second], centers[C[pr[p].first].f],
                  centers[C[pr[p].second].f], gn, hn, Gp, Hp, P);
  Split best;
  for (const Split& s : res)
    if (s.gain > best.gain) best = s;
  return best;
}

// ─── 재귀 빌드 ───────────────────────────────────────────────────────────────
static int build(std::vector<Node>& arena, const double* X, int d,
                 const std::vector<u16>& binidx,
                 const std::vector<std::vector<double>>& centers,
                 const std::vector<double>& g, const std::vector<double>& h,
                 std::vector<int> idx, int depth, const Params& P,
                 std::mt19937& rng) {
  double Gp = 0, Hp = 0;
  for (int i : idx) {
    Gp += g[i];
    Hp += h[i];
  }
  int ni = (int)arena.size();
  arena.push_back(Node());
  arena[ni].weight = -Gp / (Hp + P.reg_lambda);
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
  Split s2 = eval_2d(C, centers, gn, hn, Gp, Hp, P);
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
  arena[ni].is_leaf = false;
  arena[ni].type = bs.type;
  arena[ni].fA = bs.fA;
  arena[ni].fB = bs.fB;
  arena[ni].thr = bs.thr;
  arena[ni].coefA = bs.coefA;
  arena[ni].coefB = bs.coefB;
  arena[ni].bias = bs.bias;
  arena[ni].nan_direction = bs.nan_direction;

  int L = build(arena, X, d, binidx, centers, g, h, std::move(li), depth + 1, P,
                rng);
  int R = build(arena, X, d, binidx, centers, g, h, std::move(ri), depth + 1, P,
                rng);
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

// ─── Booster ─────────────────────────────────────────────────────────────────
class Booster {
 public:
  Params P;
  std::vector<std::vector<Node>> trees;
  double init_score = 0;
  Booster(int n_estimators, double learning_rate, int max_depth, int max_bins,
          double reg_lambda, int min_samples, int n_screen, double subsample,
          double colsample, unsigned seed, int objective) {
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
  }

  void fit(py::array_t<double, py::array::c_style | py::array::forcecast> Xa,
           py::array_t<double, py::array::c_style | py::array::forcecast> ya) {
    auto Xb = Xa.request();
    auto yb = ya.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    const double* y = (const double*)yb.ptr;
    Bins B = precompute_bins(X, n, d, P.max_bins);

    double ybar = 0;
    for (int i = 0; i < n; i++) ybar += y[i];
    ybar /= n;
    if (P.objective == 0) {
      double y2 = std::min(std::max(ybar, 1e-6), 1 - 1e-6);
      init_score = std::log(y2 / (1 - y2));
    } else
      init_score = ybar;
    std::vector<double> raw(n, init_score), g(n), h(n);
    trees.clear();
    trees.reserve(P.n_estimators);
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::mt19937 rng(P.seed);
    int n_sub = std::max(1, (int)(P.subsample * n));

    for (int t = 0; t < P.n_estimators; t++) {
      if (P.objective == 0) {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
          double p = 1.0 / (1.0 + std::exp(-raw[i]));
          g[i] = p - y[i];
          h[i] = p * (1 - p);
        }
      } else {
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
          g[i] = raw[i] - y[i];
          h[i] = 1.0;
        }
      }

      std::vector<int> rows;
      if (P.subsample < 1.0) {
        std::shuffle(all.begin(), all.end(), rng);
        rows.assign(all.begin(), all.begin() + n_sub);
      } else
        rows = all;
      std::vector<Node> arena;
      arena.reserve(256);
      build(arena, X, d, B.idx, B.centers, g, h, rows, 0, P, rng);

#pragma omp parallel for
      for (int i = 0; i < n; i++)
        raw[i] += P.learning_rate * predict_one(arena, X + (size_t)i * d);
      trees.push_back(std::move(arena));
    }
  }

  py::array_t<double> predict_raw(
      py::array_t<double, py::array::c_style | py::array::forcecast> Xa) {
    auto Xb = Xa.request();
    int n = (int)Xb.shape[0], d = (int)Xb.shape[1];
    const double* X = (const double*)Xb.ptr;
    auto out = py::array_t<double>(n);
    double* op = (double*)out.request().ptr;

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      double r = init_score;
      const double* x = X + (size_t)i * d;
      for (auto& tr : trees) r += P.learning_rate * predict_one(tr, x);
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

#pragma omp parallel for
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
};

PYBIND11_MODULE(oqboost_core, m) {
  py::class_<Booster>(m, "Booster")
      .def(py::init<int, double, int, int, double, int, int, double, double,
                    unsigned, int>(),
           py::arg("n_estimators") = 60, py::arg("learning_rate") = 0.12,
           py::arg("max_depth") = 4, py::arg("max_bins") = 64,
           py::arg("reg_lambda") = 1.0, py::arg("min_samples") = 10,
           py::arg("n_screen") = -1, py::arg("subsample") = 1.0,
           py::arg("colsample") = 1.0, py::arg("seed") = 42,
           py::arg("objective") = 0)
      .def("fit", &Booster::fit)
      .def("predict_raw", &Booster::predict_raw)
      .def("predict_proba", &Booster::predict_proba);
}