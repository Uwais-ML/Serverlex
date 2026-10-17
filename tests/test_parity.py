import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from serverlex.preprocessing.pipeline import MLPipeline
from serverlex.serving.deployer import ModelDeployer


def test_train_serve_parity_batch_vs_single_row():
    """
    CRITICAL PARITY REGRESSION TEST:
    Verifies that:
    1. Pipeline transform + model.predict(X_test)
    2. Deployer batch predict: deployer.predict(test_df)
    3. Deployer single-row predict: [deployer.predict(row.to_dict()) for row in test_df]
    ALL produce 100% identical predictions without train/serve skew.
    """
    np.random.seed(42)
    n_train = 500
    n_test = 100

    cities = ["New York", "San Francisco", "Chicago", "Miami", "Austin"]

    train_data = {
        "age": np.random.randint(18, 70, n_train),
        "bill": [f"${np.random.uniform(20, 200):.2f}" for _ in range(n_train)],
        "city": np.random.choice(cities, n_train),
        "churn": np.random.choice([0, 1], n_train)
    }
    train_df = pd.DataFrame(train_data)

    test_data = {
        "age": np.random.randint(18, 70, n_test),
        "bill": [f"${np.random.uniform(20, 200):.2f}" for _ in range(n_test)],
        # Test subset only has 2 cities to test onehot drop_first alignment
        "city": np.random.choice(["San Francisco", "Miami"], n_test),
        "churn": np.random.choice([0, 1], n_test)
    }
    test_df = pd.DataFrame(test_data)

    # 1. Fit Pipeline
    pipe = MLPipeline()
    X_train, y_train = pipe.fit_transform(
        train_df,
        target="churn",
        numeric_string_cols=["bill"],
        encode="onehot",
        scale="standard"
    )

    # 2. Train Model
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)

    # 3. Reference Predictions directly through pipeline
    X_test, _ = pipe.transform(test_df)
    ref_preds = model.predict(X_test)

    # 4. Deployer Batch Predictions
    deployer = ModelDeployer(pipeline=pipe, model=model)
    batch_preds = deployer.predict(test_df)

    # 5. Deployer Single-Row (Row-by-Row) Predictions
    single_preds = []
    for _, row in test_df.iterrows():
        row_dict = row.drop("churn").to_dict()
        pred = deployer.predict(row_dict)
        single_preds.append(pred[0])
    single_preds = np.array(single_preds)

    # All three MUST be 100% identical!
    np.testing.assert_array_equal(ref_preds, batch_preds, err_msg="Batch predictions diverge from pipeline reference!")
    np.testing.assert_array_equal(batch_preds, single_preds, err_msg="Single-row API predictions diverge from batch predictions!")


def test_categorical_nan_imputation_parity():
    """
    Verifies that NaN in categorical columns is correctly imputed with mode
    and does NOT become a literal 'nan' string.
    """
    train_df = pd.DataFrame({
        "city": ["New York", "New York", "Chicago", np.nan],
        "target": [0, 1, 0, 1]
    })
    test_df = pd.DataFrame({
        "city": [np.nan],
        "target": [0]
    })

    pipe = MLPipeline()
    pipe.fit(train_df, target="target", encode="onehot")

    # Mode of train is 'newyork'
    assert pipe.cat_fill["city"] == "newyork"

    # Transform test set with NaN
    X_test, _ = pipe.transform(test_df)

    # Onehot columns must not contain 'city_nan'
    assert not any("nan" in col for col in X_test.columns), "Found 'nan' in onehot columns! NaN was not imputed."
