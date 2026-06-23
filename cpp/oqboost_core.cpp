// oqboost_core.cpp — OQBoost 2.0 core (histogram-binned 2D-oblique GBDT)
// 2D-oblique Newton-boosted GBDT. 전역 사전 binning(히스토그램 트릭)으로
// 노드별 정렬 제거. 범주 서브시스템 없음(정수코드=연속). pybind11.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using u16 = uint16_t;

static inline double gain_term(double G, double H, double lam) { return G*G/(H+lam); }

static double percentile_sorted(const std::vector<double>& xs, double p) {
    int n=(int)xs.size(); if(n==1) return xs[0];
    double rank=p/100.0*(n-1); int lo=(int)std::floor(rank); int hi=std::min(lo+1,n-1);
    return xs[lo]+(rank-lo)*(xs[hi]-xs[lo]);
}
static std::vector<double> unique_sorted(std::vector<double> v){
    std::sort(v.begin(),v.end()); v.erase(std::unique(v.begin(),v.end()),v.end()); return v;
}

// ─── 전역 사전 binning (히스토그램 트릭의 핵심) ──────────────────────────────
//   피처별 quantile 경계를 full-X에서 1회 계산 → 샘플→bin 인덱스 미리 저장.
//   centers[f][b] = bin b의 대표 좌표(평균 raw값) — oblique 좌표로 사용.
struct Bins {
    std::vector<u16> idx;                  // n*d bin index (row-major)
    std::vector<std::vector<double>> edges;   // [f] 정렬 경계
    std::vector<std::vector<double>> centers; // [f] bin 대표값
};

static Bins precompute_bins(const double* X, int n, int d, int max_bins) {
    Bins B; B.idx.resize((size_t)n*d); B.edges.resize(d); B.centers.resize(d);
    std::vector<double> col(n);
    for (int f=0; f<d; f++) {
        for (int i=0;i<n;i++) col[i]=X[(size_t)i*d+f];
        std::vector<double> cs(col); std::sort(cs.begin(),cs.end());
        // quantile 경계 (max_bins-1개) → 중복 제거
        std::vector<double> e;
        for (int b=1;b<max_bins;b++) e.push_back(percentile_sorted(cs, 100.0*b/max_bins));
        e = unique_sorted(e);
        int nb=(int)e.size()+1;
        B.edges[f]=e;
        std::vector<double> sum(nb,0); std::vector<int> cnt(nb,0);
        for (int i=0;i<n;i++) {
            int b=(int)(std::upper_bound(e.begin(),e.end(),col[i])-e.begin());
            B.idx[(size_t)i*d+f]=(u16)b; sum[b]+=col[i]; cnt[b]++;
        }
        std::vector<double>& ctr=B.centers[f]; ctr.resize(nb);
        for (int b=0;b<nb;b++) {
            if (cnt[b]>0) ctr[b]=sum[b]/cnt[b];
            else ctr[b] = (b==0 ? (e.empty()?cs[0]:e[0])
                                : (b-1<(int)e.size()? e[b-1] : cs.back()));
        }
    }
    return B;
}

// ─── 전역 feature-feature |corr| (1회). 강상관 쌍 = oblique서 redundant ─────
static std::vector<float> feature_corr(const double* X, int n, int d) {
    std::vector<double> mean(d, 0), sd(d, 0);
    for (int f = 0; f < d; f++) {
        double m = 0; for (int i = 0; i < n; i++) m += X[(size_t)i*d+f]; m /= n;
        double v = 0; for (int i = 0; i < n; i++) { double t = X[(size_t)i*d+f]-m; v += t*t; }
        mean[f] = m; sd[f] = std::sqrt(v / n);
    }
    std::vector<float> C((size_t)d*d, 0.f);
    for (int a = 0; a < d; a++) {
        C[(size_t)a*d+a] = 1.f;
        for (int b = a+1; b < d; b++) {
            if (sd[a] < 1e-12 || sd[b] < 1e-12) continue;
            double cov = 0;
            for (int i = 0; i < n; i++) cov += (X[(size_t)i*d+a]-mean[a])*(X[(size_t)i*d+b]-mean[b]);
            float r = (float)std::fabs(cov / n / (sd[a]*sd[b]));
            C[(size_t)a*d+b] = r; C[(size_t)b*d+a] = r;
        }
    }
    return C;
}

// ─── 분할/노드 ───────────────────────────────────────────────────────────────
struct Split { double gain=0; int type=0,fA=-1,fB=-1; double thr=0,coefA=0,coefB=0,bias=0; };
struct Node {
    bool is_leaf=true; double weight=0; int type=0,fA=-1,fB=-1;
    double thr=0,coefA=0,coefB=0,bias=0; int left=-1,right=-1;
};
struct Params {
    int n_estimators=60, max_depth=4, max_bins=64, min_samples=10;
    double learning_rate=0.12, reg_lambda=1.0;
    int n_screen=-1; double subsample=1.0, colsample=1.0;
    unsigned seed=42; int objective=0;
    double corr_skip=1.01;   // |feature-corr|>=이값 쌍은 oblique서 스킵 (1.01=off)
};

// ─── SIS 스크리닝 ────────────────────────────────────────────────────────────
static std::vector<int> screen(const double* X, int d, const std::vector<int>& idx,
                               const std::vector<double>& g, int m) {
    if (m<0||m>=d){ std::vector<int> a(d); std::iota(a.begin(),a.end(),0); return a; }
    int n=(int)idx.size(); double gm=0; for(int i=0;i<n;i++) gm+=g[idx[i]]; gm/=n;
    double gvar=0; for(int i=0;i<n;i++){double t=g[idx[i]]-gm; gvar+=t*t;}
    double gstd=std::sqrt(gvar/n); std::vector<double> score(d,0);
    if (gstd>1e-12) for(int f=0;f<d;f++){
        double xm=0; for(int i=0;i<n;i++) xm+=X[(size_t)idx[i]*d+f]; xm/=n;
        double xv=0,cov=0; for(int i=0;i<n;i++){double xt=X[(size_t)idx[i]*d+f]-xm; xv+=xt*xt; cov+=xt*(g[idx[i]]-gm);}
        double xs=std::sqrt(xv/n); score[f]=(xs>1e-12)?std::fabs(cov/n/(xs*gstd)):0;
    }
    std::vector<int> fs(d); std::iota(fs.begin(),fs.end(),0);
    std::partial_sort(fs.begin(),fs.begin()+m,fs.end(),[&](int a,int b){return score[a]>score[b];});
    fs.resize(m); std::sort(fs.begin(),fs.end()); return fs;
}

// ─── H-가중 LSQ 선형 분리면 (2×2) ───────────────────────────────────────────
static bool lsq_separator(const std::vector<double>& cA, const std::vector<double>& cB,
                          const std::vector<int>& lab, const std::vector<double>& Hs,
                          double& oA, double& oB) {
    int S=(int)cA.size(); double a00=0,a01=0,a11=0,b0=0,b1=0;
    for(int i=0;i<S;i++){double H=Hs[i],u=cA[i],v=cB[i],l=lab[i];
        a00+=H*u*u;a01+=H*u*v;a11+=H*v*v;b0+=H*u*l;b1+=H*v*l;}
    double det=a00*a11-a01*a01,dA,dB;
    if(std::fabs(det)<1e-10){
        double w0=0,w1=0,m0u=0,m0v=0,m1u=0,m1v=0;
        for(int i=0;i<S;i++){ if(lab[i]==0){w0+=Hs[i];m0u+=Hs[i]*cA[i];m0v+=Hs[i]*cB[i];}
            else{w1+=Hs[i];m1u+=Hs[i]*cA[i];m1v+=Hs[i]*cB[i];}}
        if(w0<1e-10||w1<1e-10) return false; dA=m1u/w1-m0u/w0; dB=m1v/w1-m0v/w0;
    } else { dA=(a11*b0-a01*b1)/det; dB=(-a01*b0+a00*b1)/det; }
    double nrm=std::sqrt(dA*dA+dB*dB); if(nrm<1e-10) return false;
    oA=dA/nrm; oB=dB/nrm; return true;
}

// ─── 투영 위 히스토그램 임계 (O(n+B)) ───────────────────────────────────────
static bool refine_threshold(const std::vector<double>& proj, const double* gn, const double* hn,
                             double lam, double Gp, double Hp, double& outT, double& outGain) {
    int n=(int)proj.size(); double mn=proj[0],mx=proj[0];
    for(double v:proj){mn=std::min(mn,v);mx=std::max(mx,v);}
    if(mx-mn<1e-12) return false;
    const int B=64; double w=(mx-mn)/B; double Gb[64]={0},Hb[64]={0};
    for(int i=0;i<n;i++){int b=(int)((proj[i]-mn)/w); if(b>=B)b=B-1; if(b<0)b=0;
        Gb[b]+=gn[i]; Hb[b]+=hn[i];}
    double base=gain_term(Gp,Hp,lam),GL=0,HL=0,bg=0,bt=0; bool found=false;
    for(int b=0;b+1<B;b++){ GL+=Gb[b]; HL+=Hb[b];
        if(HL<=1e-12||Hp-HL<=1e-12) continue;
        double gn=gain_term(GL,HL,lam)+gain_term(Gp-GL,Hp-HL,lam)-base;
        if(gn>bg){bg=gn;bt=mn+(b+1)*w;found=true;}}
    outT=bt; outGain=bg; return found;
}

// ─── 노드 피처 캐시 (정렬 없음: bin은 사전계산, raw col만 gather) ─────────────
struct FCache { int f; std::vector<double> col; std::vector<u16> bin; };

static std::vector<FCache> build_caches(const double* X, int d, const std::vector<u16>& binidx,
        const std::vector<int>& idx, const std::vector<int>& feats) {
    std::vector<FCache> C(feats.size());
    for(size_t fi=0;fi<feats.size();fi++){
        FCache& c=C[fi]; c.f=feats[fi]; int f=c.f;
        c.col.resize(idx.size()); c.bin.resize(idx.size());
        for(size_t i=0;i<idx.size();i++){
            c.col[i]=X[(size_t)idx[i]*d+f]; c.bin[i]=binidx[(size_t)idx[i]*d+f];
        }
    }
    return C;
}

// ─── 1D 분할 (bin 히스토그램) ───────────────────────────────────────────────
static Split eval_1d(const std::vector<FCache>& C, const std::vector<std::vector<double>>& centers,
                     const std::vector<double>& gn, const std::vector<double>& hn,
                     double Gp, double Hp, const Params& P) {
    Split best; double base=gain_term(Gp,Hp,P.reg_lambda); int nloc=(int)gn.size();
    for(const FCache& c:C){
        const std::vector<double>& ctr=centers[c.f]; int k=(int)ctr.size(); if(k<2) continue;
        std::vector<double> Ga(k,0),Ha(k,0); std::vector<int> cnt(k,0);
        for(int i=0;i<nloc;i++){int b=c.bin[i]; Ga[b]+=gn[i]; Ha[b]+=hn[i]; cnt[b]++;}
        std::vector<int> occ; for(int a=0;a<k;a++) if(cnt[a]>0) occ.push_back(a);
        if((int)occ.size()<2) continue;
        std::sort(occ.begin(),occ.end(),[&](int a,int b){
            return -Ga[a]/(Ha[a]+P.reg_lambda) < -Ga[b]/(Ha[b]+P.reg_lambda);});
        double GL=0,HL=0;
        for(int ki=0;ki+1<(int)occ.size();ki++){
            GL+=Ga[occ[ki]]; HL+=Ha[occ[ki]];
            double gn=gain_term(GL,HL,P.reg_lambda)+gain_term(Gp-GL,Hp-HL,P.reg_lambda)-base;
            if(gn>best.gain){
                double lv=-1e300,rv=1e300;
                for(int j=0;j<=ki;j++) lv=std::max(lv,ctr[occ[j]]);
                for(int j=ki+1;j<(int)occ.size();j++) rv=std::min(rv,ctr[occ[j]]);
                best.gain=gn; best.type=1; best.fA=c.f; best.thr=(lv+rv)/2.0;
            }
        }
    }
    return best;
}

// ─── 한 쌍 2D oblique (bin 그리드 + 사전계산 center) ────────────────────────
static Split eval_pair(const FCache& cA_, const FCache& cB_,
                       const std::vector<double>& ctrA, const std::vector<double>& ctrB,
                       const std::vector<double>& gn, const std::vector<double>& hn,
                       double Gp, double Hp, const Params& P) {
    Split s; int kA=(int)ctrA.size(), kB=(int)ctrB.size(); if(kA<1||kB<1) return s;
    int nloc=(int)gn.size(), K=kA*kB;
    // thread_local 스크래치 — 쌍마다 heap 할당 제거 (각 OpenMP 스레드 전용)
    static thread_local std::vector<double> Gc,Hc,proj; static thread_local std::vector<int> cnt;
    Gc.assign(K,0); Hc.assign(K,0); cnt.assign(K,0);
    for(int i=0;i<nloc;i++){int c=cA_.bin[i]*kB+cB_.bin[i];
        Gc[c]+=gn[i]; Hc[c]+=hn[i]; cnt[c]++;}
    std::vector<int> oa,ob; std::vector<double> Gs,Hs;
    for(int a=0;a<kA;a++) for(int b=0;b<kB;b++){int c=a*kB+b; if(cnt[c]>0){
        oa.push_back(a);ob.push_back(b);Gs.push_back(Gc[c]);Hs.push_back(Hc[c]);}}
    int S=(int)oa.size(); if(S<2) return s;
    std::vector<int> so(S); std::iota(so.begin(),so.end(),0);
    std::sort(so.begin(),so.end(),[&](int a,int b){
        return -Gs[a]/(Hs[a]+P.reg_lambda) < -Gs[b]/(Hs[b]+P.reg_lambda);});
    double base=gain_term(Gp,Hp,P.reg_lambda),GL=0,HL=0,bg=0; int bk=-1;
    for(int ki=0;ki+1<S;ki++){GL+=Gs[so[ki]];HL+=Hs[so[ki]];
        double gv=gain_term(GL,HL,P.reg_lambda)+gain_term(Gp-GL,Hp-HL,P.reg_lambda)-base;
        if(gv>bg){bg=gv;bk=ki;}}
    if(bk<0) return s;
    std::vector<int> lab(S,1); for(int j=0;j<=bk;j++) lab[so[j]]=0;
    std::vector<double> cA(S),cB(S); for(int t=0;t<S;t++){cA[t]=ctrA[oa[t]];cB[t]=ctrB[ob[t]];}
    double coefA,coefB; if(!lsq_separator(cA,cB,lab,Hs,coefA,coefB)) return s;
    proj.resize(nloc);
    for(int i=0;i<nloc;i++) proj[i]=coefA*cA_.col[i]+coefB*cB_.col[i];
    double t,gn2; if(!refine_threshold(proj,gn.data(),hn.data(),P.reg_lambda,Gp,Hp,t,gn2)) return s;
    s.gain=gn2; s.type=2; s.fA=cA_.f; s.fB=cB_.f; s.coefA=coefA; s.coefB=coefB; s.bias=-t;
    return s;
}

static Split eval_2d(const std::vector<FCache>& C, const std::vector<std::vector<double>>& centers,
                     const std::vector<double>& gn, const std::vector<double>& hn,
                     double Gp, double Hp, const Params& P, const float* corr, int d) {
    int nf=(int)C.size(); std::vector<std::pair<int,int>> pr; pr.reserve(nf*(nf-1)/2);
    for(int a=0;a<nf;a++) for(int b=a+1;b<nf;b++){
        // 강상관 쌍은 oblique 평면이 1D로 붕괴 → redundant, 스킵
        if(corr && corr[(size_t)C[a].f*d+C[b].f] >= P.corr_skip) continue;
        pr.emplace_back(a,b);
    }
    int np=(int)pr.size(); std::vector<Split> res(np);
    #pragma omp parallel for schedule(dynamic,4)
    for(int p=0;p<np;p++)
        res[p]=eval_pair(C[pr[p].first],C[pr[p].second],
                         centers[C[pr[p].first].f],centers[C[pr[p].second].f],
                         gn,hn,Gp,Hp,P);
    Split best; for(const Split& s:res) if(s.gain>best.gain) best=s; return best;
}

// ─── 재귀 빌드 ───────────────────────────────────────────────────────────────
static int build(std::vector<Node>& arena, const double* X, int d, const std::vector<u16>& binidx,
                 const std::vector<std::vector<double>>& centers,
                 const std::vector<double>& g, const std::vector<double>& h,
                 std::vector<int> idx, int depth, const Params& P, std::mt19937& rng,
                 const float* corr) {
    double Gp=0,Hp=0; for(int i:idx){Gp+=g[i];Hp+=h[i];}
    int ni=(int)arena.size(); arena.push_back(Node()); arena[ni].weight=-Gp/(Hp+P.reg_lambda);
    if(depth>=P.max_depth||(int)idx.size()<P.min_samples) return ni;

    auto feats=screen(X,d,idx,g,P.n_screen);
    if(P.colsample<1.0&&(int)feats.size()>2){
        int keep=std::max(2,(int)std::ceil(P.colsample*feats.size()));
        std::shuffle(feats.begin(),feats.end(),rng); feats.resize(keep);
        std::sort(feats.begin(),feats.end());
    }
    auto C=build_caches(X,d,binidx,idx,feats);
    // 노드-로컬 연속 g/h (쌍·피처마다 random gather 제거)
    std::vector<double> gn(idx.size()), hn(idx.size());
    for(size_t i=0;i<idx.size();i++){gn[i]=g[idx[i]]; hn[i]=h[idx[i]];}
    Split s1=eval_1d(C,centers,gn,hn,Gp,Hp,P);
    Split s2=eval_2d(C,centers,gn,hn,Gp,Hp,P,corr,d);
    Split bs=(s2.gain>=s1.gain)?s2:s1;
    if(bs.gain<=1e-6||bs.type==0) return ni;

    std::vector<int> li,ri;
    for(int i:idx){ bool left;
        if(bs.type==1) left=X[(size_t)i*d+bs.fA]<bs.thr;
        else{double sc=bs.coefA*X[(size_t)i*d+bs.fA]+bs.coefB*X[(size_t)i*d+bs.fB]+bs.bias; left=sc<0;}
        (left?li:ri).push_back(i);
    }
    if(li.empty()||ri.empty()) return ni;
    arena[ni].is_leaf=false; arena[ni].type=bs.type; arena[ni].fA=bs.fA; arena[ni].fB=bs.fB;
    arena[ni].thr=bs.thr; arena[ni].coefA=bs.coefA; arena[ni].coefB=bs.coefB; arena[ni].bias=bs.bias;
    int L=build(arena,X,d,binidx,centers,g,h,std::move(li),depth+1,P,rng,corr);
    int R=build(arena,X,d,binidx,centers,g,h,std::move(ri),depth+1,P,rng,corr);
    arena[ni].left=L; arena[ni].right=R; return ni;
}

static inline double predict_one(const std::vector<Node>& A, const double* x){
    int ni=0; while(true){const Node& nd=A[ni]; if(nd.is_leaf) return nd.weight;
        int ch; if(nd.type==1) ch=(x[nd.fA]<nd.thr)?0:1;
        else{double s=nd.coefA*x[nd.fA]+nd.coefB*x[nd.fB]+nd.bias; ch=(s<0)?0:1;}
        ni=(ch==0)?nd.left:nd.right;}
}

// ─── Booster ─────────────────────────────────────────────────────────────────
class Booster {
public:
    Params P; std::vector<std::vector<Node>> trees; double init_score=0;
    Booster(int n_estimators,double learning_rate,int max_depth,int max_bins,double reg_lambda,
            int min_samples,int n_screen,double subsample,double colsample,unsigned seed,int objective,
            double corr_skip){
        P.n_estimators=n_estimators;P.learning_rate=learning_rate;P.max_depth=max_depth;
        P.max_bins=max_bins;P.reg_lambda=reg_lambda;P.min_samples=min_samples;P.n_screen=n_screen;
        P.subsample=subsample;P.colsample=colsample;P.seed=seed;P.objective=objective;
        P.corr_skip=corr_skip;
    }

    void fit(py::array_t<double,py::array::c_style|py::array::forcecast> Xa,
             py::array_t<double,py::array::c_style|py::array::forcecast> ya){
        auto Xb=Xa.request(); auto yb=ya.request();
        int n=(int)Xb.shape[0], d=(int)Xb.shape[1];
        const double* X=(const double*)Xb.ptr; const double* y=(const double*)yb.ptr;
        Bins B=precompute_bins(X,n,d,P.max_bins);    // 히스토그램 binning 1회
        std::vector<float> corr;                     // 강상관 쌍 스킵용 (1회)
        if(P.corr_skip<=1.0) corr=feature_corr(X,n,d);
        const float* corrp = corr.empty()?nullptr:corr.data();

        double ybar=0; for(int i=0;i<n;i++) ybar+=y[i]; ybar/=n;
        if(P.objective==0){double y2=std::min(std::max(ybar,1e-6),1-1e-6); init_score=std::log(y2/(1-y2));}
        else init_score=ybar;
        std::vector<double> raw(n,init_score),g(n),h(n);
        trees.clear(); trees.reserve(P.n_estimators);
        std::vector<int> all(n); std::iota(all.begin(),all.end(),0);
        std::mt19937 rng(P.seed); int n_sub=std::max(1,(int)(P.subsample*n));
        for(int t=0;t<P.n_estimators;t++){
            if(P.objective==0) for(int i=0;i<n;i++){double p=1.0/(1.0+std::exp(-raw[i])); g[i]=p-y[i]; h[i]=p*(1-p);}
            else for(int i=0;i<n;i++){g[i]=raw[i]-y[i]; h[i]=1.0;}
            std::vector<int> rows;
            if(P.subsample<1.0){std::shuffle(all.begin(),all.end(),rng); rows.assign(all.begin(),all.begin()+n_sub);}
            else rows=all;
            std::vector<Node> arena; arena.reserve(256);
            build(arena,X,d,B.idx,B.centers,g,h,rows,0,P,rng,corrp);
            for(int i=0;i<n;i++) raw[i]+=P.learning_rate*predict_one(arena,X+(size_t)i*d);
            trees.push_back(std::move(arena));
        }
    }

    py::array_t<double> predict_raw(py::array_t<double,py::array::c_style|py::array::forcecast> Xa){
        auto Xb=Xa.request(); int n=(int)Xb.shape[0],d=(int)Xb.shape[1];
        const double* X=(const double*)Xb.ptr; auto out=py::array_t<double>(n);
        double* op=(double*)out.request().ptr;
        for(int i=0;i<n;i++){double r=init_score; const double* x=X+(size_t)i*d;
            for(auto& tr:trees) r+=P.learning_rate*predict_one(tr,x); op[i]=r;}
        return out;
    }
    py::array_t<double> predict_proba(py::array_t<double,py::array::c_style|py::array::forcecast> Xa){
        auto Xb=Xa.request(); int n=(int)Xb.shape[0],d=(int)Xb.shape[1];
        const double* X=(const double*)Xb.ptr;
        auto out=py::array_t<double>({(py::ssize_t)n,(py::ssize_t)2});
        double* op=(double*)out.request().ptr;
        for(int i=0;i<n;i++){double r=init_score; const double* x=X+(size_t)i*d;
            for(auto& tr:trees) r+=P.learning_rate*predict_one(tr,x);
            double p=1.0/(1.0+std::exp(-r)); op[i*2]=1-p; op[i*2+1]=p;}
        return out;
    }
};

PYBIND11_MODULE(oqboost_core, m) {
    py::class_<Booster>(m,"Booster")
        .def(py::init<int,double,int,int,double,int,int,double,double,unsigned,int,double>(),
             py::arg("n_estimators")=60, py::arg("learning_rate")=0.12, py::arg("max_depth")=4,
             py::arg("max_bins")=64, py::arg("reg_lambda")=1.0, py::arg("min_samples")=10,
             py::arg("n_screen")=-1, py::arg("subsample")=1.0, py::arg("colsample")=1.0,
             py::arg("seed")=42, py::arg("objective")=0, py::arg("corr_skip")=1.01)
        .def("fit",&Booster::fit)
        .def("predict_raw",&Booster::predict_raw)
        .def("predict_proba",&Booster::predict_proba);
}
