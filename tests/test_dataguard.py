import pytest
import pandas as pd
import numpy as np
from serverlex.diagnostics.dataguard import DataGuard


def test_target_leakage_detection():
    np.random.seed(42)
    target = np.random.choice([0, 1], 200)
    # 0.92 correlation feature
    leaking_feat = target * 2.0 + np.random.normal(0, 0.4, 200)

    df = pd.DataFrame({
        "legit_feature": np.random.randn(200),
        "leaking_feature": leaking_feat,
        "target": target
    })

    guard = DataGuard(df, target="target")
    leakage = guard.check_target_leakage()
    assert len(leakage) == 1
    assert leakage[0][0] == "leaking_feature"
    assert leakage[0][1] >= 0.90

    # Test in audit report
    report = guard.run_audit()
    assert "Target Leakage" in report["Check"].values


def test_unique_id_column_detection():
    df = pd.DataFrame({
        "account_id": [f"acc_{i}" for i in range(100)],
        "val": range(100)
    })
    guard = DataGuard(df)
    ids = guard.check_id_columns()
    assert "account_id" in ids


def test_infinite_values_detection():
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.inf, -np.inf, 5.0],
        "cat": ["a", "b", "c", "d", "e"]
    })
    guard = DataGuard(df)
    infs = guard.check_infinite_values()
    assert len(infs) == 1
    assert infs[0][0] == "num"
    assert infs[0][1] == 2


def test_high_cardinality_detection():
    df = pd.DataFrame({
        "city_zip": [f"zip_{i}" for i in range(150)],
        "val": range(150)
    })
    guard = DataGuard(df)
    high_cards = guard.check_cardinality(threshold=100)
    assert len(high_cards) == 1
    assert high_cards[0][0] == "city_zip"
