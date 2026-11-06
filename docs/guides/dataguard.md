# Pre-Flight Data Auditing with `DataGuard` 🛡️

`serverlex.DataGuard` is a pre-flight safety auditor that detects structural defects in tabular datasets before model training begins.

---

## 🔍 What `DataGuard` Catches

1. **Unique ID Columns**: Columns with 100% uniqueness (or $\ge 0.95$ combined with ID naming patterns like `id`, `uuid`, `key`, `ssn`) that would cause memorization.
2. **Target Leakage**: Features correlating $|r| \ge 0.90$ with the target variable, which usually indicates data collected post-event.
3. **PII Columns**: High-cardinality personal identifiers (`name`, `email`, `phone`).
4. **Infinite Values**: Columns containing `+inf` or `-inf`.
5. **Collinear Clusters**: Highly correlated feature pairs ($|r| \ge 0.85$).
6. **Class Imbalance**: Severe distribution skews ($< 10\%$ minority representation).

---

## 💻 Usage Examples

### Running an Audit

```python
import pandas as pd
from serverlex import DataGuard

df = pd.read_csv("customers.csv")
guard = DataGuard(df, target="churn")

# Run full diagnostic audit
report = guard.run_audit(leakage_threshold=0.90, id_threshold=0.95)

# Inspect findings
print("Flagged ID columns:", guard.id_cols)
print("Flagged Leakage columns:", guard.leakage_cols)
print("Flagged PII columns:", guard.pii_cols)
print("Collinear Pairs:", guard.collinear_pairs)
```

### Exporting an Interactive HTML Dashboard

Generate a self-contained, dark-mode dashboard with dataset health grades (A+ to F):

```python
guard.to_html_report("audit_dashboard.html")
```

### Auto-Remediation

Automatically clean the dataset in one call:

```python
clean_df, summary = guard.auto_remediate(
    target="churn",
    drop_ids=True,
    drop_leakage=True,
    drop_pii=True
)

print(f"Dropped {summary['total_dropped']} columns: {summary['dropped_columns']}")
print("Breakdown by category:", summary['dropped_by_category'])
print(f"Features: {summary['initial_columns']} -> {summary['remaining_columns']}")
```
