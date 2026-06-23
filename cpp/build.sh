#!/bin/bash
# OQBoost 2.0 C++ 모듈 빌드 (cmake 불필요, clang 직접). 개발용 in-place 빌드.
# 배포 빌드는 `pip install .` (setup.py가 동일 확장 컴파일).
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PB11=$($PY -c "import pybind11; print(pybind11.get_include())")
EXT=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# OpenMP (libomp via homebrew). 없으면 OMP 플래그 빼고 단일스레드로.
OMP_PREFIX=$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
OMP_FLAGS=""
if [ -f "$OMP_PREFIX/lib/libomp.dylib" ]; then
    OMP_FLAGS="-Xpreprocessor -fopenmp -I$OMP_PREFIX/include -L$OMP_PREFIX/lib -lomp"
    echo "OpenMP: $OMP_PREFIX"
fi

c++ -O3 -std=c++17 -shared -undefined dynamic_lookup -fPIC \
    -I"$PYINC" -I"$PB11" $OMP_FLAGS \
    oqboost_core.cpp -o "../oqboost/oqboost_core${EXT}"

echo "built -> oqboost/oqboost_core${EXT}"
