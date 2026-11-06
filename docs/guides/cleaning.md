# Data Sanitization with `Clean` 🧼

`serverlex.Clean` cleans dirty tabular data without corrupting valid features.

---

## 🔑 Key Capabilities

- **Header Normalization**: Auto-strips embedded headers and normalizes casing.
- **Dirty Currency Strings**: Coerces `$1,250.00` and `€45.90` to pure `float64`.
- **Outlier Handling**: Robust IQR outlier clipping without altering binary indicator columns or target variables.
- **Missing Value Imputation**: Flexible strategies: `'md'` (median), `'m'` (mean), `'z'` (zero), `'f'` (forward fill), `'b'` (backward fill).

---

## 💻 Usage Examples

### Column Type Separation

```python
from serverlex import Clean

cleaner = Clean(df)
dtypes = cleaner.separate()

print("Numeric columns:", dtypes["numeric"])
print("Categorical columns:", dtypes["categorical"])
print("Datetime columns:", dtypes["datetime"])
```

### Total Clean

```python
cleaned_df = cleaner.total_clean(
    numeric_string_cols=["monthly_bill", "total_paid"], # Strips $, commas -> float
    missing_param="md",                                  # Median imputation
    outlier_method="IQR",                                # 1.5 * IQR clamping
    null_threshold=0.70                                  # Drops columns with > 70% missing
)
```

### Domain Range Guards (Best Practice)

When data contains impossible entries (e.g. `Age = 220`), filter them before running `total_clean()` to prevent skewing the median:

```python
# Convert impossible domain entries to NaN
df.loc[~df["Age"].between(18, 90), "Age"] = float("nan")

# Clean normally with median fill
clean_df = Clean(df).total_clean(missing_param="md")
```
