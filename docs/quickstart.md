# Quickstart ⚡

Get started with Serverlex in under 60 seconds.

---

## 🏃 Complete End-to-End Workflow

This complete script demonstrates reading messy data, auditing it with `DataGuard`, fitting an `MLPipeline`, evaluating a model, and exporting a secured FastAPI serving microservice.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from serverlex import (
    DataGuard,
    MLPipeline,
    ModelEvaluator,
    ModelDeployer
)

# 1. Load your raw dataset
df = pd.read_csv("customers.csv")

# 2. Automated Quality Audit
guard = DataGuard(df, target="churn")
guard.run_audit()

# Auto-remediate flagged ID columns, PII, and leakage
clean_df, summary = guard.auto_remediate(
    target="churn",
    drop_ids=True,
    drop_leakage=True,
    drop_pii=True
)
print("Dropped problematic columns:", summary["dropped_columns"])

# 3. Train/Test Split
train_df = clean_df.iloc[:800].copy()
test_df = clean_df.iloc[800:].copy()

# 4. Leakage-Free Preprocessing Pipeline
pipe = MLPipeline()
X_train, y_train = pipe.fit_transform(
    train_df,
    target="churn",
    encode="onehot",
    scale="standard",
    data_time=["signup_date", "%Y-%m-%d"],
    numeric_string_cols=["monthly_bill"]  # Parses currency like '$89.50'
)
X_test, y_test = pipe.transform(test_df)

# 5. Train & Evaluate Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

evaluator = ModelEvaluator(model=model, X_test=X_test, y_test=y_test)
metrics = evaluator.evaluate()
print(f"Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

# 6. Save Model Bundle with HMAC Authentication
deployer = ModelDeployer(pipeline=pipe, model=model)
deployer.save("churn_bundle.joblib", compute_hash=True, hmac_key="production_secret_key")

# 7. Predict directly on a single raw JSON record!
sample_input = {
    "signup_date": "2024-05-15",
    "monthly_bill": "$92.40",
    "device": "laptop",
    "tenure_months": 14,
    "support_tickets": 2
}
prediction = deployer.predict(sample_input, strict=True)
print("Prediction:", prediction[0])
```

---

## 🚀 Exporting to a Live Microservice

To turn this into a live REST API microservice:

```python
deployer.export_api(
    output_dir="./serving_service",
    bundle_filename="churn_bundle.joblib",
    hmac_key="production_secret_key",
    host="0.0.0.0",
    port=8000
)
```

Run the service:

```bash
cd serving_service
export SERVERLEX_HMAC_KEY="production_secret_key"
uvicorn app:app --host 0.0.0.0 --port 8000
```
