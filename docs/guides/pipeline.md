# Leakage-Free Pipeline with `MLPipeline` ⚙️

`serverlex.MLPipeline` encapsulates feature engineering, encoding, and scaling to ensure 100% train/serve parity with zero data leakage.

---

## 🔒 The Leakage-Free Guarantee

In standard Scikit-Learn workflows, beginners often accidentally fit scalers or imputers on the full dataset before splitting, which leaks test distribution statistics into the training set.

`MLPipeline` guarantees:
1. All parameters (scaling means, standard deviations, categorical levels, median fills) are **only** computed during `.fit()` on training data.
2. In `.transform()`, those frozen parameters are applied directly to test rows or single-record production payloads.

---

## 💻 Usage Examples

### Fitting and Transforming

```python
from serverlex import MLPipeline

pipe = MLPipeline()

# Fit on training split
X_train, y_train = pipe.fit_transform(
    train_df,
    target="churn",
    encode="onehot",                         # 'onehot', 'label', 'target', or 'none'
    scale="standard",                        # 'standard', 'minmax', 'robust', or 'none'
    drop_pii=True,                           # Automatically drops high-cardinality PII
    auto_remediate=True,                     # Automatically excludes ID columns & leakage proxies
    data_time=["signup_date", "%Y-%m-%d"],   # Expands to year, month, day, dayofweek
    numeric_string_cols=["monthly_bill"]     # Parses currency strings
)

# Transform test split with exact column parity
X_test, y_test = pipe.transform(test_df)

assert list(X_train.columns) == list(X_test.columns)
assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0
```

### Baseline Model Leaderboard

Run multi-model 5-fold cross-validation in one line:

```python
leaderboard = pipe.evaluate_baseline_models(cv=5)
print(leaderboard)
```
