import os
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if os.path.exists("README.md") else "End-to-End ML Preprocessing, Diagnostics, and Serving Platform"

setup(
    name="serverlex",
    version="1.0.1",
    author="Serverlex Team",
    description="End-to-End Automated Data Cleaning, ML Preprocessing, Diagnostics, and Deployment Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["serverlex"],
    py_modules=[
        "Clean",
        "Visualization",
        "DataCheck",
        "Pipeline",
        "FeatureSelector",
        "ModelHyperparameter",
        "EnsembleBuilder",
        "Evaluation",
        "Explainability",
        "ModelDeployer",
        "DriftDetector",
    ],
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
)
