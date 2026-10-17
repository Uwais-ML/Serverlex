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
    deployer.save(bundle_path, compute_hash=True)
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
    deployer = ModelDeployer.load(trained_bundle)

    # Payload with missing 'income', rogue key 'device_ip', and string 'age'
    faulty_payload = {
        "age": "32",
        "city": "Chicago",
        "device_ip": "192.168.1.1"
    }

    pred = deployer.predict(faulty_payload)
    assert len(pred) == 1
    assert pred[0] in [0, 1]


def test_deployer_strict_mode(trained_bundle):
    deployer = ModelDeployer.load(trained_bundle)
    # Missing 'city' in strict mode must raise ValueError
    incomplete_payload = {"age": 30.0, "income": 50000.0}
    with pytest.raises(ValueError, match="Strict deployment error"):
        deployer.predict(incomplete_payload, strict=True)


def test_deployer_bundle_integrity(trained_bundle):
    # Load with verify_hash=True must succeed on untampered bundle
    d = ModelDeployer.load(trained_bundle, verify_hash=True)
    assert d.pipeline is not None

    # Tampering with file must fail integrity verification
    with open(trained_bundle, "ab") as f:
        f.write(b"tamper_bytes")

    with pytest.raises(ValueError, match="integrity verification failed"):
        ModelDeployer.load(trained_bundle, verify_hash=True)


def test_deployer_batch_prediction(trained_bundle):
    deployer = ModelDeployer.load(trained_bundle)
    batch = [
        {"age": 22.0, "income": 30000.0, "city": "New York"},
        {"age": 55.0, "income": 120000.0, "city": "Chicago"}
    ]
    preds = deployer.predict(batch)
    assert len(preds) == 2
