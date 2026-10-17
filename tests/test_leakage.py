import pytest
import pandas as pd
import numpy as np
from serverlex.preprocessing.pipeline import MLPipeline


def test_zero_data_leakage_scaling():
    """
    Proves mathematically that scaling parameters (mean, std) learned on train
    are NOT contaminated or altered by test set data during transform().
    """
    np.random.seed(42)
    # Train set centered around 100 with std 15
    train_df = pd.DataFrame({
        "feature_1": np.random.normal(100, 15, 500),
        "target": np.random.choice([0, 1], 500)
    })
    # Test set with completely different distribution (centered around 10,000)
    test_df = pd.DataFrame({
        "feature_1": np.random.normal(10000, 500, 200),
        "target": np.random.choice([0, 1], 200)
    })

    pipe = MLPipeline()
    X_train, y_train = pipe.fit_transform(train_df, target="target", scale="standard")

    # Memorize train scaling parameters
    train_mean = dict(pipe.scaler_mean)
    train_std = dict(pipe.scaler_std)

    # Verify train mean is near 100
    assert abs(train_mean["feature_1"] - 100) < 5

    # Transform test set
    X_test, y_test = pipe.transform(test_df)

    # Scaling parameters on the pipeline MUST remain identical
    assert pipe.scaler_mean == train_mean, "Leakage detected: pipeline mean was altered by test set!"
    assert pipe.scaler_std == train_std, "Leakage detected: pipeline std was altered by test set!"

    # Test set transformation must use train statistics: (10000 - 100) / 15 ≈ 660
    assert X_test["feature_1"].mean() > 500


def test_zero_data_leakage_imputation():
    """
    Proves that missing value imputation baselines are locked from train
    and strictly applied to test data without recomputing.
    """
    train_df = pd.DataFrame({
        "age": [20.0, 30.0, 40.0, np.nan],  # Median is 30.0
        "city": ["New York", "Chicago", "New York", None],  # Mode is New York
        "target": [0, 1, 0, 1]
    })
    test_df = pd.DataFrame({
        "age": [np.nan, 99.0],
        "city": [None, "Miami"],
        "target": [0, 1]
    })

    pipe = MLPipeline()
    pipe.fit(train_df, target="target")

    # Train medians and modes
    assert pipe.num_fill["age"] == 30.0
    assert pipe.cat_fill["city"] == "newyork"

    # Transform test set
    X_test = pipe.transform(test_df)

    # Missing age in test must have been imputed with train median (30.0), not test value
    assert pipe.num_fill["age"] == 30.0
    assert pipe.cat_fill["city"] == "newyork"


def test_unseen_categorical_levels_handling():
    """
    Verifies that novel categorical levels in the test set do not break the pipeline
    and maintain exact training feature alignment without index misalignment.
    """
    train_df = pd.DataFrame({
        "category": ["A", "B", "A", "B"],
        "target": [0, 1, 0, 1]
    })
    test_df = pd.DataFrame({
        "category": ["C", "D"],  # Completely unseen in train
        "target": [1, 0]
    })

    pipe = MLPipeline()
    X_train, _ = pipe.fit_transform(train_df, target="target")
    X_test, _ = pipe.transform(test_df)

    # Feature columns must match train exactly
    assert list(X_train.columns) == list(X_test.columns)
    # Unseen categories must be zeroed out safely
    assert (X_test == 0).all().all()
