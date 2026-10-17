import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from serverlex.diagnostics.dataguard import DataGuard
from serverlex.diagnostics.drift import DriftDetector
from serverlex.modeling.selector import FeatureSelector
from serverlex.preprocessing.pipeline import MLPipeline
from serverlex.serving.deployer import ModelDeployer


def test_dataguard_auto_remediate():
    """Verify DataGuard.auto_remediate correctly drops IDs, PII, and leakage columns."""
    n = 60
    np.random.seed(42)
    churn = np.array([0, 1] * (n // 2))
    df = pd.DataFrame({
        "cust_id": [f"ID_{i:04d}" for i in range(n)],
        "full_name": [f"User Name {i}" for i in range(n)],
        "churn_score": churn + np.random.normal(0, 0.01, n),  # correlation ~0.99 with target
        "monthly_fee": np.linspace(20, 100, n),
        "tenure": np.random.randint(1, 24, n),
        "churn": churn
    })

    guard = DataGuard(df, target="churn")
    clean_df, summary = guard.auto_remediate(drop_ids=True, drop_pii=True, drop_leakage=True)

    assert "cust_id" not in clean_df.columns
    assert "full_name" not in clean_df.columns
    assert "churn_score" not in clean_df.columns
    assert "monthly_fee" in clean_df.columns
    assert "tenure" in clean_df.columns
    assert "churn" in clean_df.columns
    assert summary["remaining_columns"] == 3


def test_dataguard_check_collinearity():
    """Verify DataGuard flags pairwise collinear features."""
    n = 50
    x1 = np.linspace(10, 100, n)
    x2 = x1 * 2.0 + np.random.normal(0, 0.001, n)  # r = 1.0
    x3 = np.random.normal(0, 1, n)

    df = pd.DataFrame({"feat_a": x1, "feat_b": x2, "feat_c": x3})
    guard = DataGuard(df)
    collinear = guard.check_collinearity(threshold=0.90)

    assert len(collinear) >= 1
    pair_names = {(c[0], c[1]) for c in collinear}
    assert ("feat_a", "feat_b") in pair_names or ("feat_b", "feat_a") in pair_names


def test_dataguard_to_html_report(tmp_path):
    """Verify DataGuard generates a standalone valid HTML report."""
    df = pd.DataFrame({
        "cust_id": [f"CUST_{i}" for i in range(30)],
        "email": [f"user{i}@example.com" for i in range(30)],
        "age": np.random.randint(18, 70, 30),
        "target": [0, 1] * 15
    })
    guard = DataGuard(df, target="target")
    report_file = str(tmp_path / "data_health.html")
    html_content = guard.to_html_report(filepath=report_file, title="Test Audit Report")

    assert os.path.exists(report_file)
    assert "<!DOCTYPE html>" in html_content
    assert "Health Score" in html_content
    assert "cust_id" in html_content
    assert "Diagnostic Quality" in html_content


def test_feature_selector_prune_collinear():
    """Verify FeatureSelector.prune_collinear drops redundant collinear partners."""
    n = 100
    np.random.seed(42)
    y = np.array([0, 1] * (n // 2))

    feat_predictive = y + np.random.normal(0, 0.1, n)
    feat_collinear = feat_predictive + np.random.normal(0, 0.001, n)  # r ~ 1.0 with predictive
    feat_independent = np.random.normal(0, 1, n)

    df = pd.DataFrame({
        "f_predictive": feat_predictive,
        "f_collinear": feat_collinear,
        "f_independent": feat_independent
    })

    # When y is passed, it should retain the one with higher predictive power
    pruned_df, dropped = FeatureSelector.prune_collinear(df, threshold=0.90, y=y)
    assert len(dropped) == 1
    assert "f_independent" in pruned_df.columns
    assert len(pruned_df.columns) == 2


def test_pipeline_fit_auto_remediate():
    """Verify MLPipeline.fit(auto_remediate=True) drops leakage, PII, and ID columns."""
    n = 60
    np.random.seed(42)
    churn = np.array([0, 1] * (n // 2))
    df = pd.DataFrame({
        "customer_id": [f"ID_{i:04d}" for i in range(n)],
        "full_name": [f"Customer {i}" for i in range(n)],
        "churn_proxy": churn * 0.99,  # leakage
        "tenure": np.random.randint(1, 24, n),
        "churn": churn
    })

    pipe = MLPipeline()
    pipe.fit(df, target="churn", auto_remediate=True)

    # Remediated columns must not appear in feature columns
    assert not any("customer_id" in c for c in pipe.feature_columns)
    assert not any("full_name" in c for c in pipe.feature_columns)
    assert not any("churn_proxy" in c for c in pipe.feature_columns)
    assert "tenure" in pipe.feature_columns

    # Transform on test row should also cleanly drop them
    test_row = {
        "customer_id": "ID_9999",
        "full_name": "New Customer",
        "churn_proxy": 0.85,
        "tenure": 12
    }
    X = pipe.transform(test_row)
    assert X.shape[1] == len(pipe.feature_columns)
    assert "churn_proxy" not in X.columns


def test_deployer_benchmark(tmp_path):
    """Verify ModelDeployer.benchmark calculates latency percentiles and throughput."""
    df = pd.DataFrame({
        "tenure": [1, 12, 24, 6] * 10,
        "fee": [10.0, 50.0, 80.0, 30.0] * 10,
        "churn": [0, 1, 0, 1] * 10
    })
    pipe = MLPipeline()
    pipe.fit(df, target="churn")
    X, y = pipe.transform(df)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

    deployer = ModelDeployer(pipeline=pipe, model=model)
    stats = deployer.benchmark(n_iterations=50, warmup=10)

    assert "p50_ms" in stats
    assert "p95_ms" in stats
    assert "p99_ms" in stats
    assert "throughput_req_per_sec" in stats
    assert stats["iterations"] == 50
    assert stats["p50_ms"] >= 0.0


def test_deployer_export_api(tmp_path):
    """Verify ModelDeployer.export_api generates app.py, Dockerfile, and requirements.txt."""
    df = pd.DataFrame({
        "tenure": [1, 12, 24, 6] * 10,
        "fee": [10.0, 50.0, 80.0, 30.0] * 10,
        "churn": [0, 1, 0, 1] * 10
    })
    pipe = MLPipeline()
    pipe.fit(df, target="churn")
    X, y = pipe.transform(df)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

    deployer = ModelDeployer(pipeline=pipe, model=model)
    export_dir = str(tmp_path / "fastapi_service")
    res = deployer.export_api(output_dir=export_dir, hmac_key="test_secret_key")

    assert os.path.exists(res["app"])
    assert os.path.exists(res["dockerfile"])
    assert os.path.exists(res["requirements"])
    assert os.path.exists(res["bundle"])
    assert os.path.exists(res["readme"])

    # Verify app.py contains FastAPI definitions and BUNDLE_FILE
    with open(res["app"], "r", encoding="utf-8") as f:
        app_code = f.read()
    assert "FastAPI" in app_code
    assert "/predict" in app_code
    assert "/health" in app_code
    assert "test_secret_key" in app_code


def test_drift_detector_raise_on_drift():
    """Verify DriftDetector raises RuntimeError when raise_on_drift=True and drift is detected."""
    np.random.seed(42)
    ref = pd.DataFrame({"feature_1": np.random.normal(0, 1, 100)})
    shifted = pd.DataFrame({"feature_1": np.random.normal(10, 1, 100)})  # huge shift

    detector = DriftDetector(ref, alpha=0.01, psi_threshold=0.15)
    with pytest.raises(RuntimeError, match="Critical data drift detected"):
        detector.detect_drift(shifted, raise_on_drift=True)
