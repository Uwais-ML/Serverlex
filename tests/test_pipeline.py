import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from serverlex.preprocessing.pipeline import MLPipeline


def test_pipeline_fit_transform_and_baseline_evaluation():
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 65, 100),
        "income": np.random.normal(50000, 10000, 100),
        "city": np.random.choice(["NYC", "LA", "CHI"], 100),
        "signup_date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "churn": np.random.choice([0, 1], 100)
    })

    pipe = MLPipeline()
    X_train, y_train = pipe.fit_transform(
        df,
        target="churn",
        data_time=["signup_date"],
        encode="onehot",
        scale="standard"
    )

    # Verify datetime extraction
    assert "signup_date_year" in X_train.columns
    assert "signup_date_month" in X_train.columns
    assert "signup_date_day" in X_train.columns

    # Add custom model
    pipe.add_model("custom_rf", RandomForestClassifier(n_estimators=5, random_state=42))

    # Evaluate baseline models without passing arguments
    leaderboard_auto = pipe.evaluate_baseline_models(cv=3)
    assert isinstance(leaderboard_auto, pd.DataFrame)
    assert "custom_rf" in leaderboard_auto["Model"].values

    # Evaluate baseline models with explicit (X, y) arguments
    leaderboard_explicit = pipe.evaluate_baseline_models(X_train, y_train, cv=3)
    assert isinstance(leaderboard_explicit, pd.DataFrame)
