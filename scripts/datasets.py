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
