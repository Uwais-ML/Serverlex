import pytest
import pandas as pd
import numpy as np
from serverlex.diagnostics.drift import DriftDetector


def test_numerical_drift_detection():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "income": np.random.normal(50000, 10000, 300)
    })
    # Shifted production data
    prod_df = pd.DataFrame({
        "income": np.random.normal(90000, 15000, 300)
    })

    detector = DriftDetector(ref_df, alpha=0.05, psi_threshold=0.2)
    report = detector.detect_drift(prod_df)

    assert len(report) == 1
    row = report.iloc[0]
    assert row["Feature"] == "income"
    assert row["Drift_Detected"] is True or row["Drift_Detected"] == True
    assert row["PSI"] > 0.25
    assert row["Wasserstein"] > 10000


def test_categorical_drift_detection():
    ref_df = pd.DataFrame({
        "category": ["standard"] * 90 + ["premium"] * 10
    })
    # Shifted distribution
    prod_df = pd.DataFrame({
        "category": ["standard"] * 10 + ["premium"] * 90
    })

    detector = DriftDetector(ref_df, alpha=0.05)
    report = detector.detect_drift(prod_df)

    assert len(report) == 1
    row = report.iloc[0]
    assert row["Feature"] == "category"
    assert row["Drift_Detected"] is True or row["Drift_Detected"] == True


def test_no_drift_when_distributions_match():
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "val": np.random.normal(100, 10, 500)
    })
    prod_df = pd.DataFrame({
        "val": np.random.normal(100, 10, 500)
    })

    detector = DriftDetector(ref_df, alpha=0.01, psi_threshold=0.2)
    report = detector.detect_drift(prod_df)

    assert len(report) == 1
    row = report.iloc[0]
    assert row["Drift_Detected"] is False or row["Drift_Detected"] == False
    assert row["PSI"] < 0.1
