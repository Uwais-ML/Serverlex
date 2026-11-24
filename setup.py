import os
import sys
import logging
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if os.path.exists("README.md") else "End-to-End ML Preprocessing, Diagnostics, and Serving Platform"

# Compiler flags for C++17 native acceleration
if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2", "/EHsc", "/GR-"]
    extra_link_args = []
else:
    extra_compile_args = ["-std=c++17", "-O3", "-ffast-math", "-funroll-loops"]
    extra_link_args = []


class Pybind11Include:
    """Helper to delay pybind11 import until setup_requires/build-system has installed it."""
    def __str__(self):
        import pybind11
        return pybind11.get_include()


class OptionalBuildExt(build_ext):
    """
    Build C++ extensions if compiler & pybind11 are available.
    If compilation fails on the user's system, fall back gracefully to pure Python
    so that `pip install` always succeeds.
    """
    def run(self):
        try:
            super().run()
        except Exception as e:
            logging.warning("⚠️  Could not compile C++ extension (_serverlex_ext): %s", e)
            logging.warning("⚠️  Falling back to pure Python implementation. Serverlex will still work normally.")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            logging.warning("⚠️  Failed to build extension '%s': %s", ext.name, e)
            logging.warning("⚠️  Falling back to pure Python implementation.")


ext_modules = [
    Extension(
        name="serverlex._serverlex_ext",
        sources=["serverlex/_ext/psi_kernel.cpp"],
        include_dirs=[Pybind11Include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name="serverlex",
    version="2.1.0",
    author="Serverlex Team",
    description="Production-Safe ML Preprocessing, Diagnostics, Tuning, and Serving Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "serverlex": ["*.so", "*.pyd", "_ext/*"],
    },
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalBuildExt},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "copilot": ["torch>=2.0.0", "transformers>=4.40.0", "accelerate>=0.26.0"],
        "fast": ["pybind11>=2.10.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.4.0",
            "twine>=5.0",
            "build>=1.0",
        ],
        "all": [
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "accelerate>=0.26.0",
            "pybind11>=2.10.0",
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.4.0",
            "twine>=5.0",
            "build>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "serverlex-copilot=serverlex.copilot.cli:main"
        ]
    },
)
