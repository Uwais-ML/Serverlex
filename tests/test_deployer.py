import os
import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from serverlex.preprocessing.pipeline import MLPipeline
from serverlex.serving.deployer import ModelDeployer


@pytest.fixture
def trained_bundle(tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({
        "age": [25.0, 40.0, 35.0, 50.0],
        "income": [45000.0, 80000.0, 60000.0, 95000.0],
        "city": ["New York", "Chicago", "Miami", "Chicago"],
        "churn": [0, 1, 0, 1]
    })
    pipe = MLPipeline()
    X_t, y_t = pipe.fit_transform(df, target="churn")
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_t, y_t)

    bundle_path = str(tmp_path / "bundle.joblib")
    deployer = ModelDeployer(pipeline=pipe, model=model)
    deployer.save(bundle_path)
    return bundle_path


def test_deployer_loading_patterns(trained_bundle):
    # Pattern 1: Classmethod
    d1 = ModelDeployer.load(trained_bundle)
    assert d1.pipeline is not None
    assert d1.model is not None

    # Pattern 2: Instance load
    d2 = ModelDeployer()
    d2.load(trained_bundle)
    assert d2.pipeline is not None
    assert d2.model is not None

    # Pattern 3: Direct constructor
    d3 = ModelDeployer(trained_bundle)
    assert d3.pipeline is not None
    assert d3.model is not None


def test_deployer_single_record_prediction(trained_bundle):
    deployer = ModelDeployer.load(trained_bundle)
    record = {"age": 30.0, "income": 55000.0, "city": "Miami"}
    pred = deployer.predict(record)
    assert len(pred) == 1
    assert pred[0] in [0, 1]


def test_deployer_schema_resilience(trained_bundle):
    """
    Tests that missing keys in production payload are auto-imputed from training baselines,
    rogue unmodeled keys are filtered, and string numbers are coerced.
    """
    deployer = ModelDeployer.load(trained_bundle)

    # Payload with missing 'income', rogue key 'device_ip', and string 'age'
    faulty_payload = {
        "age": "32",  # String number needing coercion
        "city": "Chicago",
        "device_ip": "192.168.1.1"  # Rogue key not in training
        # 'income' is omitted intentionally
    }

    pred = deployer.predict(faulty_payload)
    assert len(pred) == 1
    assert pred[0] in [0, 1]


def test_deployer_batch_prediction(trained_bundle):
    deployer = ModelDeployer.load(trained_bundle)
    batch = [
        {"age": 22.0, "income": 30000.0, "city": "New York"},
        {"age": 55.0, "income": 120000.0, "city": "Chicago"}
    ]
    preds = deployer.predict(batch)
    assert len(preds) == 2
