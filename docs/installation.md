# Installation

```bash
pip install oqboost
```

Optional extras:

```bash
pip install oqboost[plot]    # matplotlib, for oqboost.plot
pip install oqboost[bench]   # xgboost / lightgbm / catboost / pandas / matplotlib (benchmarks)
```

## Prebuilt wheels

Wheels are published for CPython 3.10–3.13 on:

- Linux (manylinux_2_28, x86_64)
- Windows (amd64)
- macOS (arm64)

## Building from source

On other platforms `pip` builds from the sdist. You need:

- a C++17 compiler (`clang++` or `g++`)
- OpenMP for parallel training (`brew install libomp` on macOS; bundled with GCC on Linux)

```bash
git clone https://github.com/cree1116/oqboost-2.0
cd oqboost-2.0
pip install -e .
```

To skip OpenMP (single-threaded build): `OQBOOST_NO_OPENMP=1 pip install -e .`

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24, scikit-learn ≥ 1.3
