import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from serverlex.preprocessing.clean import Clean
from serverlex.preprocessing.pipeline import MLPipeline
from serverlex.serving.deployer import ModelDeployer
from serverlex.diagnostics.drift import DriftDetector
from serverlex.diagnostics.evaluation import ModelEvaluator


def test_outlier_cell_wise_preserves_target():
    """Bug 1 Regression: Outlier handling must not touch target column or overwrite entire rows."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "num1": np.random.normal(10, 2, n),
        "num2": np.random.normal(50, 5, n),
        "churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    })
    # Inject 5 extreme outliers into num1 only
    df.loc[:4, "num1"] = 9999.0

    cleaner = Clean(df)
    cleaned = cleaner.total_clean(outlier_method="IQR", outlier_param="m", corr_target="churn")

    # Target column must remain strictly binary 0 and 1
    unique_churn = set(cleaned["churn"].unique())
    assert unique_churn.issubset({0, 1}), f"Target churn corrupted: {unique_churn}"

    # num2 in rows 0-4 should NOT be altered to the column mean because num2 was not an outlier!
    orig_num2_first5 = df.loc[:4, "num2"].values
    cleaned_num2_first5 = cleaned.loc[:4, "num2"].values
    np.testing.assert_allclose(cleaned_num2_first5, orig_num2_first5, rtol=1e-5)


def test_outlier_cell_wise_only_modifies_offending_cells():
    """Bug 1 Regression: Outlier clipping is cell-wise, leaving normal cells in that row unchanged."""
    df = pd.DataFrame({
        "col_a": [10.0, 12.0, 14.0, 16.0, 1000.0],  # index 4 is extreme outlier
        "col_b": [25.0, 25.0, 25.0, 25.0, 25.0]     # index 4 is normal
    })
    cleaner = Clean(df)
    cleaner.outliers_IQR(param='m')

    # col_b at index 4 must remain 25.0
    assert cleaner.data.loc[4, "col_b"] == 25.0
    # col_a at index 4 should be imputed
    assert cleaner.data.loc[4, "col_a"] < 1000.0


def test_scaler_bounds_on_training_minmax():
    """Issue 3 Regression: minmax scaler must bound training features cleanly in [0, 1]."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feat1": np.random.uniform(10, 100, n),
        "feat2": np.random.normal(0, 1, n),
        "target": np.random.choice([0, 1], n)
    })
    pipeline = MLPipeline()
    X_train, y_train = pipeline.fit_transform(df, target="target", scale="minmax", outlier_method=None)

    for col in X_train.columns:
        col_min = X_train[col].min()
        col_max = X_train[col].max()
        assert col_min >= -1e-6, f"MinMax lower bound violated on {col}: {col_min}"
        assert col_max <= 1.0 + 1e-6, f"MinMax upper bound violated on {col}: {col_max}"


def test_strict_mode_rejects_unseen_categories():
    """Bug 2 Regression: strict=True must reject unseen categorical levels instead of silent mapping."""
    train_df = pd.DataFrame({
        "city": ["new_york", "chicago", "san_francisco"] * 20,
        "amount": [100.0, 200.0, 300.0] * 20,
        "target": [0, 1, 0] * 20
    })
    pipeline = MLPipeline()
    X, y = pipeline.fit_transform(train_df, target="target", scale=None)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    deployer = ModelDeployer(pipeline=pipeline, model=model)

    # In strict mode, an unseen city 'kolkata' must raise ValueError
    with pytest.raises(ValueError, match="unseen categories"):
        deployer.predict({"city": "kolkata", "amount": 150.0}, strict=True)


def test_strict_mode_rejects_non_numeric_values():
    """Bug 2 Regression: strict=True must reject non-numeric strings in numeric features."""
    train_df = pd.DataFrame({
        "age": [25.0, 30.0, 35.0, 40.0] * 10,
        "target": [0, 1, 0, 1] * 10
    })
    pipeline = MLPipeline()
    X, y = pipeline.fit_transform(train_df, target="target", scale=None)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    deployer = ModelDeployer(pipeline=pipeline, model=model)

    # In strict mode, age="abc" must raise ValueError
    with pytest.raises(ValueError, match="non-numeric"):
        deployer.predict({"age": "abc"}, strict=True)


def test_deployer_corrupt_bundle_raises_clean_error(tmp_path):
    """Issue 6 Regression: Corrupt bundle load must raise clear ValueError."""
    corrupt_file = str(tmp_path / "corrupt_bundle.joblib")
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_JOBLIB_PICKLE_FILE_CONTENT_123456789")

    with pytest.raises(ValueError, match="Corrupt or invalid model bundle"):
        ModelDeployer.load(corrupt_file)


def test_deployer_hmac_authenticity(tmp_path):
    """Issue 6 Feature: HMAC bundle signing and verification."""
    pipeline = MLPipeline()
    model = RandomForestClassifier()
    deployer = ModelDeployer(pipeline=pipeline, model=model)

    bundle_path = str(tmp_path / "secure_bundle.joblib")
    secret_key = "super_secret_production_key"

    deployer.save(bundle_path, hmac_key=secret_key)
    assert os.path.exists(bundle_path + ".hmac")

    # Load with correct key succeeds
    loaded = ModelDeployer.load(bundle_path, hmac_key=secret_key)
    assert loaded.pipeline is not None

    # Load with wrong key fails
    with pytest.raises(ValueError, match="HMAC mismatch"):
        ModelDeployer.load(bundle_path, hmac_key="wrong_key")


def test_drift_tiny_batch_guard():
    """Minor Issue Regression: Tiny batches (n=1, n=5) must not trigger false positive categorical drift."""
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "category": ["A", "B", "C"] * 100,
        "value": np.random.normal(10, 2, 300)
    })
    detector = DriftDetector(ref_df)

    tiny_cur = pd.DataFrame({
        "category": ["A"],
        "value": [10.2]
    })
    report = detector.detect_drift(tiny_cur)
    cat_row = report[report["Feature"] == "category"].iloc[0]

    assert not cat_row["Drift_Detected"], "False positive categorical drift triggered on n=1"
    assert "Insufficient Samples" in cat_row["Severity"]


def test_drift_constant_feature_shift():
    """Minor Issue Regression: Constant feature changing in current data must be flagged as drift."""
    ref_df = pd.DataFrame({
        "sensor_mode": [1] * 100
    })
    detector = DriftDetector(ref_df)

    # Production data changed to constant 2
    cur_df = pd.DataFrame({
        "sensor_mode": [2] * 50
    })
    report = detector.detect_drift(cur_df)
    row = report[report["Feature"] == "sensor_mode"].iloc[0]

    assert bool(row["Drift_Detected"]) is True
    assert row["Type"] == "Constant Shift"


def test_model_evaluator_returns_metrics_dict():
    """Minor Issue Regression: ModelEvaluator.evaluate() must return structured metrics dictionary."""
    evaluator = ModelEvaluator()

    X = np.random.normal(0, 1, (100, 3))
    y = np.random.choice([0, 1], size=100)

    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    metrics = evaluator.evaluate(model, X, y)

    assert isinstance(metrics, dict), f"Expected dict return, got {type(metrics)}"
    assert metrics["task"] == "Classification"
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics


def test_outlier_preserves_binary_and_degenerate_columns():
    """1.2.2 Critical Bug 1 Regression: Binary and degenerate IQR columns must not be flattened."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "continuous": np.random.normal(100, 15, n),
        "has_partner": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),  # Q1=0, Q3=0, IQR=0
        "is_senior": np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    })
    # Inject one genuine outlier into continuous only
    df.loc[0, "continuous"] = 9999.0

    cleaner = Clean(df)
    cleaned = cleaner.total_clean(outlier_method="IQR", outlier_param="m")

    # Both binary columns must remain strictly {0, 1}
    assert set(cleaned["has_partner"].unique()).issubset({0, 1}), f"has_partner corrupted: {cleaned['has_partner'].unique()}"
    assert set(cleaned["is_senior"].unique()).issubset({0, 1}), f"is_senior corrupted: {cleaned['is_senior'].unique()}"
    # The genuine outlier in continuous was cleaned
    assert cleaned.loc[0, "continuous"] < 9999.0


def test_small_batch_drift_no_false_alarms():
    """1.2.2 Critical Bug 2 Regression: Small batches (n=30) from same distribution must not false alarm."""
    np.random.seed(42)
    ref_df = pd.DataFrame({
        "tenure": np.random.exponential(scale=20, size=1000),
        "monthly": np.random.normal(70, 20, size=1000)
    })
    detector = DriftDetector(ref_df)

    # 10 random draws of n=30 from the exact same distribution
    false_alarm_count = 0
    for seed in range(10):
        np.random.seed(100 + seed)
        cur_df = pd.DataFrame({
            "tenure": np.random.exponential(scale=20, size=30),
            "monthly": np.random.normal(70, 20, size=30)
        })
        report = detector.detect_drift(cur_df)
        if report["Drift_Detected"].any():
            false_alarm_count += 1

    # Should have 0 false alarms on n=30
    assert false_alarm_count == 0, f"False alarms observed on same-distribution batches: {false_alarm_count}/10"


def test_deployer_hmac_missing_key_clear_error(tmp_path):
    """1.2.2 Minor Polish Regression: Loading an HMAC bundle without key raises clear message."""
    pipeline = MLPipeline()
    model = RandomForestClassifier()
    deployer = ModelDeployer(pipeline=pipeline, model=model)

    bundle_path = str(tmp_path / "protected_bundle.joblib")
    deployer.save(bundle_path, hmac_key="secret")

    with pytest.raises(ValueError, match="saved with HMAC authenticity protection"):
        ModelDeployer.load(bundle_path)


def test_clean_drops_embedded_header_rows():
    """1.2.3 Regression: Embedded duplicate header rows from CSV concatenation are dropped."""
    df = pd.DataFrame({
        "age": ["25", "30", "age", "45"],
        "fee": ["100", "200", "fee", "300"],
        "plan": ["basic", "standard", "plan", "premium"]
    })
    cleaner = Clean(df)
    cleaner.drop_embedded_headers()
    assert len(cleaner.data) == 3
    assert "age" not in cleaner.data["age"].values


def test_pipeline_auto_coerces_numeric_strings_and_serving(tmp_path):
    """1.2.3 Regression: Columns with units/symbols (9.6 GB, 0,0) are auto-coerced to numeric and deploy cleanly."""
    df = pd.DataFrame({
        "signup_date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"] * 10,
        "usage_gb": ["6.5", "9.6 GB", "0,0", "12.2"] * 10,
        "fee_per_gb": ["45.5", "50.0", "10.2", "88.1"] * 10,
        "auto_renew": ["0", "1", "0", "1"] * 10,
        "churn": [0, 1, 0, 1] * 10
    })
    pipeline = MLPipeline()
    pipeline.fit(df, target="churn", data_time=["signup_date"], encode="onehot", scale="standard")

    # usage_gb and fee_per_gb should be numeric, not one-hot encoded
    assert "usage_gb" in pipeline.num_fill
    assert "fee_per_gb" in pipeline.num_fill
    assert "usage_gb" not in pipeline.cat_categories

    X, y = pipeline.transform(df)
    assert X.shape[1] < 15  # not blown up into dozens of categories

    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    deployer = ModelDeployer(pipeline=pipeline, model=model)
    bundle_file = str(tmp_path / "auto_bundle.joblib")
    deployer.save(bundle_file)

    loaded = ModelDeployer.load(bundle_file)
    sample = {
        "signup_date": "2024-05-01",
        "usage_gb": 10.5,
        "fee_per_gb": 47.92,
        "auto_renew": 1
    }
    pred = loaded.predict(sample, strict=True)
    assert len(pred) == 1

