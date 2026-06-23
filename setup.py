"""
setup.py — OQBoost 2.0 빌드. pybind11 확장(oqboost.oqboost_core)을 컴파일한다.
OpenMP: macOS=libomp(brew), Linux=-fopenmp(libgomp), Windows=/openmp.
없으면 단일스레드로 폴백.
"""
import platform
import subprocess
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

system = platform.system()
extra_compile, extra_link = [], []

if system == "Windows":
    extra_compile = ["/O2", "/openmp"]
elif system == "Darwin":
    extra_compile = ["-O3"]
    try:
        omp = subprocess.check_output(
            ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
        ).decode().strip()
        extra_compile += ["-Xpreprocessor", "-fopenmp", f"-I{omp}/include"]
        extra_link += [f"-L{omp}/lib", "-lomp"]
    except Exception:
        pass  # libomp 없으면 단일스레드
else:  # Linux / others
    extra_compile = ["-O3", "-fopenmp"]
    extra_link = ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "oqboost.oqboost_core",
        ["cpp/oqboost_core.cpp"],
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
        cxx_std=17,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
