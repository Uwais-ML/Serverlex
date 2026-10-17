import os
import sys
import logging
import kagglehub
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from serverlex import DataGuard, MLPipeline, ModelDeployer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Benchmark")


def run_churn_benchmark():
    logger.info("=== Running Customer Churn Benchmark ===")
    churn_path = kagglehub.dataset_download("muhammadshahidazeem/customer-churn-dataset")
    csv_file = [os.path.join(churn_path, f) for f in os.listdir(churn_path) if f.endswith(".csv")][0]
    df = pd.read_csv(csv_file)
    logger.info("Loaded churn dataset with shape: %s", df.shape)

    # 1. DataGuard Audit
    guard = DataGuard(df, target="Churn")
    report = guard.run_audit()
    logger.info("DataGuard flagged IDs: %s", guard.id_cols)

    # 2. Train/Test Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Churn"].dropna())

    # 3. Fit Pipeline
    pipe = MLPipeline()
    X_train, y_train = pipe.fit_transform(train_df, target="Churn", scale="robust")
    X_test, y_test = pipe.transform(test_df)

    # 4. Train Gradient Boosting
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info("Customer Churn Final Test Accuracy: %.4f (%.2f%%)", acc, acc * 100)
    return acc


def run_walmart_benchmark():
    logger.info("=== Running Walmart Store Sales Benchmark ===")
    walmart_path = kagglehub.dataset_download("mikhail1681/walmart-sales")
    csv_file = [os.path.join(walmart_path, f) for f in os.listdir(walmart_path) if f.endswith(".csv")][0]
    df = pd.read_csv(csv_file)
    logger.info("Loaded walmart sales dataset with shape: %s", df.shape)

    # Train/Test Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Fit Pipeline with datetime extraction
    pipe = MLPipeline()
    X_train, y_train = pipe.fit_transform(
        train_df,
        target="Weekly_Sales",
        data_time=["Date"],
        scale="standard"
    )
    X_test, y_test = pipe.transform(test_df)

    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    logger.info("Walmart Sales Test R2 Score: %.4f, RMSE: %.2f", r2, rmse)
    return r2, rmse


if __name__ == "__main__":
    run_churn_benchmark()
    run_walmart_benchmark()
