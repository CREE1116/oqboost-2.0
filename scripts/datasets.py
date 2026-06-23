"""
datasets.py — 벤치마크용 데이터셋 (합성 2D + 실제 OpenML)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml, make_moons, make_circles
from sklearn.preprocessing import OrdinalEncoder


# ══════════════════════════════════════════════════════════════════════════════
#  합성 2D 데이터 (decision boundary 시각화용)
# ══════════════════════════════════════════════════════════════════════════════

def ds_xor(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    p = 1 / (1 + np.exp(-12 * X[:, 0] * X[:, 1]))
    return X, (rng.uniform(size=n) < p).astype(int)


def ds_spiral(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    m = n // 2
    t = np.sqrt(rng.uniform(0, 1, m)) * 3 * np.pi
    def arm(t, sign):
        r = t
        return np.c_[sign * r * np.cos(t), sign * r * np.sin(t)]
    X = np.vstack([arm(t, 1), arm(t, -1)])
    X += rng.normal(0, 0.6, X.shape)
    X /= np.abs(X).max()
    y = np.r_[np.zeros(m), np.ones(m)].astype(int)
    return X, y


def ds_moons(n=1500, seed=2):
    X, y = make_moons(n_samples=n, noise=0.25, random_state=seed)
    X = (X - X.mean(0)) / X.std(0)
    return X, y.astype(int)


def ds_circles(n=1500, seed=3):
    X, y = make_circles(n_samples=n, noise=0.12, factor=0.45, random_state=seed)
    return X, y.astype(int)


def ds_checkerboard(n=1800, seed=4, k=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, k, (n, 2))
    y = ((X[:, 0].astype(int) + X[:, 1].astype(int)) % 2).astype(int)
    y[rng.uniform(size=n) < 0.05] ^= 1
    X = (X - X.mean(0)) / X.std(0)
    return X, y


def ds_gauss_quantiles(n=1500, seed=5):
    from sklearn.datasets import make_gaussian_quantiles
    X, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2, random_state=seed)
    X = (X - X.mean(0)) / X.std(0)
    return X, y.astype(int)


SYNTH_2D = {
    "XOR": ds_xor, "Spiral": ds_spiral, "Moons": ds_moons,
    "Circles": ds_circles, "Checkerboard": ds_checkerboard,
    "GaussQuantiles": ds_gauss_quantiles,
}


# ══════════════════════════════════════════════════════════════════════════════
#  실제 데이터 (OpenML, 이진 분류)
# ══════════════════════════════════════════════════════════════════════════════

def _subsample(X, y, max_rows=3000, seed=42):
    if len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), max_rows, replace=False)
    return X[idx], y[idx]


def _encode(df):
    import pandas as pd
    X = np.zeros((len(df), df.shape[1]), dtype=float)
    for j, col in enumerate(df.columns):
        s = df[col]
        if s.dtype.name in ("category", "object") or not np.issubdtype(s.dtype, np.number):
            X[:, j] = OrdinalEncoder().fit_transform(s.astype(str).values.reshape(-1, 1)).ravel()
        else:
            X[:, j] = pd.to_numeric(s, errors="coerce").fillna(0).values
    return X


def _load_openml_one(did, name, max_rows):
    data = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df = data.frame
    tcol = data.target_names[0]
    yraw = df[tcol].astype(str)
    classes = sorted(yraw.unique())
    if len(classes) != 2:
        raise ValueError(f"not binary ({len(classes)} classes)")
    y = (yraw == classes[-1]).astype(int).values
    X = np.nan_to_num(_encode(df.drop(columns=[tcol])))
    X, y = _subsample(X, y, max_rows)
    return name, X, np.asarray(y)


def _load_openml_multi(did, name, max_rows):
    """다중클래스: 클래스 수 제한 없음. y는 0..K-1 정수 코드."""
    data = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df = data.frame
    tcol = data.target_names[0]
    yraw = df[tcol].astype(str)
    classes = sorted(yraw.unique())
    if len(classes) < 3:
        raise ValueError(f"not multiclass ({len(classes)} classes)")
    code = {c: i for i, c in enumerate(classes)}
    y = yraw.map(code).values.astype(int)
    X = np.nan_to_num(_encode(df.drop(columns=[tcol])))
    X, y = _subsample(X, y, max_rows)
    return name, X, np.asarray(y)


def _load_openml_reg(did, name, max_rows):
    """회귀: 수치형 타깃. 표준화 안 함(모델이 스케일 불변)."""
    data = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df = data.frame
    tcol = data.target_names[0]
    y = np.asarray(pd_to_num(df[tcol]), dtype=float)
    X = np.nan_to_num(_encode(df.drop(columns=[tcol])))
    m = np.isfinite(y)
    X, y = X[m], y[m]
    X, y = _subsample(X, y, max_rows)
    return name, X, y


def pd_to_num(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


def load_real(max_rows=3000):
    """소형 스모크 세트 (5개)."""
    out = []
    d = load_breast_cancer()
    out.append(("breast_cancer", d.data.astype(float), d.target.astype(int)))
    for did, name in [(31, "german_credit"), (1590, "adult_income"),
                      (1461, "bank_marketing"), (37, "diabetes_pima")]:
        try:
            out.append(_load_openml_one(did, name, max_rows))
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    return out


# 다양한 이진분류 OpenML 데이터셋 (크기·차원·도메인 다양)
OPENML_SUITE = [
    (31, "german_credit"), (37, "diabetes"), (44, "spambase"),
    (1461, "bank_marketing"), (1590, "adult"), (1489, "phoneme"),
    (1462, "banknote"), (1464, "blood_transfusion"), (1494, "qsar_biodeg"),
    (1067, "kc1"), (3, "kr_vs_kp"), (4534, "phishing"),
    (1471, "eeg_eye_state"), (1487, "ozone"), (1480, "ilpd"),
    (1063, "kc2"), (1050, "pc3"), (1068, "pc1"), (40701, "churn"),
    (1504, "steel_plates_bin"), (1494, "qsar"), (151, "electricity"),
    (40983, "wilt"), (1046, "mozilla4"), (1049, "pc4"),
]

# 다중클래스 (클래스 수·차원·도메인 다양)
OPENML_MULTICLASS = [
    (54, "vehicle"),          # 4cls d18
    (23, "cmc"),              # 3cls d9
    (188, "eucalyptus"),      # 5cls d19
    (181, "yeast"),           # 10cls d8
    (40670, "dna"),           # 3cls d180
    (40982, "steel_plates"),  # 7cls d27
    (1497, "wall_robot"),     # 4cls d24
    (12, "mfeat_factors"),    # 10cls d216
    (1468, "cnae9"),          # 9cls d856 (고차원)
    (458, "analcatdata_auth"),# 4cls
    (60, "waveform"),         # 3cls d40
    (1481, "kr_vs_k"),        # 18cls
]

# 회귀 (sklearn 내장 + OpenML delve/friedman 등 다양)
OPENML_REGRESSION = [
    (537, "houses"),       # d8
    (564, "fried"),        # friedman d10
    (227, "cpu_small"),    # d12
    (574, "house_16H"),    # d16
    (308, "puma32H"),      # d32
    (296, "ailerons"),     # d40
    (215, "2dplanes"),     # d10
    (197, "cpu_act"),      # d21
    (344, "mv"),           # d10 (혼합형)
    (4544, "music_origin"),# d116 고차원
]


def load_openml_suite(max_rows=8000, include_breast=True):
    """다양한 OpenML 이진 데이터셋. 반환: list of (name, X, y)."""
    out = []
    if include_breast:
        d = load_breast_cancer()
        out.append(("breast_cancer", d.data.astype(float), d.target.astype(int)))
    for did, name in OPENML_SUITE:
        try:
            out.append(_load_openml_one(did, name, max_rows))
        except Exception as e:
            print(f"  [skip] {name}({did}): {e}")
    return out


def load_multiclass_suite(max_rows=8000, include_builtin=True):
    """다중클래스 데이터셋. 반환: list of (name, X, y)."""
    out = []
    if include_builtin:
        from sklearn.datasets import load_digits, load_wine
        dg = load_digits(); out.append(("digits", dg.data.astype(float), dg.target.astype(int)))
        wn = load_wine();   out.append(("wine", wn.data.astype(float), wn.target.astype(int)))
    for did, name in OPENML_MULTICLASS:
        try:
            out.append(_load_openml_multi(did, name, max_rows))
        except Exception as e:
            print(f"  [skip] {name}({did}): {e}")
    return out


def load_regression_suite(max_rows=8000, include_builtin=True):
    """회귀 데이터셋. 반환: list of (name, X, y)."""
    out = []
    if include_builtin:
        from sklearn.datasets import load_diabetes, fetch_california_housing
        db = load_diabetes(); out.append(("diabetes_reg", db.data.astype(float), db.target.astype(float)))
        try:
            ca = fetch_california_housing()
            X, y = _subsample(ca.data.astype(float), ca.target.astype(float), max_rows)
            out.append(("california", X, y))
        except Exception as e:
            print(f"  [skip] california: {e}")
    for did, name in OPENML_REGRESSION:
        try:
            out.append(_load_openml_reg(did, name, max_rows))
        except Exception as e:
            print(f"  [skip] {name}({did}): {e}")
    return out


# task별 loader 레지스트리 — optimize/benchmark가 참조.
SUITES = {
    "binary":     load_openml_suite,
    "multiclass": load_multiclass_suite,
    "regression": load_regression_suite,
}


def load_tasks(tasks=("binary", "multiclass", "regression"), max_rows=8000):
    """선택 task들의 데이터셋을 (name, X, y, task)로 태깅해 반환."""
    out = []
    for t in tasks:
        for name, X, y in SUITES[t](max_rows=max_rows):
            out.append((name, X, y, t))
    return out
