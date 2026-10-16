import os
import pandas as pd
import numpy as np

from DataCheck import DataGuard
from Clean import Clean
from Pipeline import MLPipeline
from ModelHyperparameter import ModelHyperparameter
from Evaluation import ModelEvaluator
from Explainability import ModelExplainer
from ModelDeployer import ModelDeployer


def test_customer_churn():
    print("\n" + "#" * 60)
    print("### TEST 1: CUSTOMER CHURN DATASET (CLASSIFICATION) ###")
    print("#" * 60 + "\n")

    train_path = "/Users/apple/.cache/kagglehub/datasets/muhammadshahidazeem/customer-churn-dataset/versions/1/customer_churn_dataset-training-master.csv"
    test_path = "/Users/apple/.cache/kagglehub/datasets/muhammadshahidazeem/customer-churn-dataset/versions/1/customer_churn_dataset-testing-master.csv"

    # Sample 5,000 rows for fast demonstration
    raw_train = pd.read_csv(train_path).dropna().sample(5000, random_state=42)
    raw_test = pd.read_csv(test_path).dropna().sample(1000, random_state=42)

    # 1. Data Quality Guardrails
    print(">>> 1. Running DataGuard on Churn...")
    guard = DataGuard(raw_train)
    guard.run_audit(target="Churn")

    # Drop CustomerID based on DataGuard recommendation
    raw_train = raw_train.drop(columns=["CustomerID"])
    raw_test = raw_test.drop(columns=["CustomerID"])

    # 2. Pipeline Fit & Transform
    print("\n>>> 2. Running MLPipeline (fit on train, transform on test)...")
    pipeline = MLPipeline()
    pipeline.fit(
        raw_train,
        target="Churn",
        encode="onehot",
        scale="standard",
        missing_param="m"
    )

    X_train, y_train = pipeline.transform(raw_train)
    X_test, y_test = pipeline.transform(raw_test)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 3. Baseline Model Evaluation
    print("\n>>> 3. Evaluating Baseline Models...")
    leaderboard = pipeline.evaluate_baseline_models(X_train, y_train, cv=3)

    # 4. Hyperparameter Tuning
    print("\n>>> 4. Hyperparameter Tuning on Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    tuner = ModelHyperparameter()
    param_grid = {"n_estimators": [30, 60], "max_depth": [5, 10]}
    tune_res = tuner.tune_non_nested(RandomForestClassifier(random_state=42), param_grid, X_train, y_train, cv=3)
    best_model = tune_res["best_estimator"]

    # 5. Deep Evaluation
    print("\n>>> 5. Deep Diagnostics on Test Set...")
    evaluator = ModelEvaluator()
    evaluator.evaluate(best_model, X_test, y_test)

    # 6. Feature Importance
    print("\n>>> 6. Model Explainability...")
    explainer = ModelExplainer(best_model, feature_names=X_train.columns)
    importances = explainer.get_feature_importances()
    print("Top 5 Driving Features:")
    print(importances.head(5).to_string(index=False))

    # 7. Model Deployment & Raw Inference
    print("\n>>> 7. Production ModelDeployer Bundle & Raw Prediction...")
    deployer = ModelDeployer(pipeline=pipeline, model=best_model)
    deployer.save("churn_bundle.joblib")

    loaded_deployer = ModelDeployer.load("churn_bundle.joblib")
    sample_customer = raw_test.iloc[0].to_dict()
    sample_customer.pop("Churn", None)

    pred = loaded_deployer.predict(sample_customer)
    prob = loaded_deployer.predict_proba(sample_customer)
    print(f"Sample Customer Prediction: {pred[0]} (Probabilities: {prob[0]})")

    # Clean up bundle file
    if os.path.exists("churn_bundle.joblib"):
        os.remove("churn_bundle.joblib")


def test_walmart_sales():
    print("\n" + "#" * 60)
    print("### TEST 2: WALMART SALES DATASET (REGRESSION & DATES) ###")
    print("#" * 60 + "\n")

    walmart_path = "/Users/apple/.cache/kagglehub/datasets/mikhail1681/walmart-sales/versions/2/Walmart_Sales.csv"
    walmart_df = pd.read_csv(walmart_path)

    train_df = walmart_df.iloc[:5000].copy()
    test_df = walmart_df.iloc[5000:].copy()

    # 1. Data Quality Guardrails
    print(">>> 1. Running DataGuard on Walmart Sales...")
    guard = DataGuard(train_df)
    guard.run_audit(target="Weekly_Sales")

    # 2. Pipeline Fit with DateTime Extraction
    print("\n>>> 2. Running MLPipeline with DateTime Extraction...")
    pipeline = MLPipeline()
    pipeline.fit(
        train_df,
        target="Weekly_Sales",
        data_time=["Date", "%d-%m-%Y"],
        scale="standard"
    )

    X_train, y_train = pipeline.transform(train_df)
    X_test, y_test = pipeline.transform(test_df)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print("Extracted Features:", X_train.columns.tolist())

    # 3. Baseline Model Evaluation (Auto detects Regression!)
    print("\n>>> 3. Evaluating Baseline Models (Regression)...")
    leaderboard = pipeline.evaluate_baseline_models(X_train, y_train, cv=3)

    # 4. Deep Regression Evaluation
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    print("\n>>> 4. Deep Regression Diagnostics on Test Set...")
    evaluator = ModelEvaluator()
    evaluator.evaluate(rf, X_test, y_test)


if __name__ == "__main__":
    test_customer_churn()
    test_walmart_sales()
