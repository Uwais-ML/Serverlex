"""
setup_ext.py — Serverlex C++ extension builder (setuptools shortcut)
─────────────────────────────────────────────────────────────────────────────
Build and install the PSI native kernel in one command:

    python setup_ext.py build_ext --inplace

Requirements:
    pip install pybind11 setuptools

The compiled .so / .pyd will land at:
    serverlex/_serverlex_ext.<platform-tag>.so   (Linux / macOS)
    serverlex/_serverlex_ext.<platform-tag>.pyd  (Windows)

After building, Serverlex automatically picks it up — no manual copy needed.
─────────────────────────────────────────────────────────────────────────────
"""
import os
import subprocess
import sys
import tempfile

from setuptools import Extension, setup

# ── Require pybind11 ─────────────────────────────────────────────────────────
try:
    import pybind11
except ImportError:
    sys.exit(
        "\n[setup_ext] pybind11 is not installed.\n"
        "Run:  pip install pybind11\n"
        "Then: python setup_ext.py build_ext --inplace\n"
    )

# ── Platform compiler flags ───────────────────────────────────────────────────
if sys.platform == "win32":
    base_compile = ["/std:c++17", "/O2", "/EHsc", "/GR-"]
    base_link    = []
    omp_compile  = ["/openmp"]
    omp_link     = []
else:
    base_compile = ["-std=c++17", "-O3", "-ffast-math", "-funroll-loops"]
    base_link    = []
    # -march=native is skipped on macOS because the universal2 target
    # (arm64 + x86_64 fat binary) is incompatible with it.
    # On Linux CI / bare non-universal builds, uncomment the line below:
    # base_compile.append("-march=native")
    omp_compile  = ["-fopenmp"]
    omp_link     = ["-fopenmp"]


# ── OpenMP probe ─────────────────────────────────────────────────────────────
def _probe_openmp() -> bool:
    """Return True if the system C++ compiler supports -fopenmp."""
    if sys.platform == "win32":
        return False          # MSVC OpenMP works differently; skip for now
    src = "#include <omp.h>\nint main(){return omp_get_max_threads();}\n"
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write(src)
        src_path = f.name
    out_path = src_path.replace(".cpp", ".out")
    try:
        r = subprocess.run(
            ["c++", "-fopenmp", src_path, "-o", out_path],
            capture_output=True,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(src_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# ── Assemble flags ────────────────────────────────────────────────────────────
compile_args = list(base_compile)
link_args    = list(base_link)

if _probe_openmp():
    print("[setup_ext] ✅ OpenMP detected — parallel PSI enabled")
    compile_args.extend(omp_compile)
    link_args.extend(omp_link)
else:
    print("[setup_ext] ⚠️  OpenMP not found — PSI will run single-threaded (still fast!)")

# ── Extension definition ──────────────────────────────────────────────────────
ext = Extension(
    name="serverlex._serverlex_ext",
    sources=["serverlex/_ext/psi_kernel.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
setup(
    name="serverlex-native",
    version="1.0.0",
    description="Serverlex C++ native extension (PSI kernel)",
    ext_modules=[ext],
    # Explicitly declare packages so setuptools does not scan the repo
    packages=["serverlex"],
    zip_safe=False,
)
