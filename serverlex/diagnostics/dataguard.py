import logging
import pandas as pd
import numpy as np
import re

logger=logging.getLogger("DataGuard")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class DataGuard:
    def __init__(self, df=None, target=None):
        if df is not None:
            self.data=df.copy()
            # Automatically strip embedded header rows (e.g. from CSV concatenation)
            min_match = max(2, len(self.data.columns) // 2)
            header_mask = (self.data == self.data.columns).sum(axis=1) >= min_match
            if header_mask.any():
                self.data = self.data[~header_mask].reset_index(drop=True)
        else:
            self.data=None
        self.target=target
        self.warnings=[]
        self.id_cols=[]
        self.leakage_cols=[]
        self.high_cardinality_cols=[]

    def set_data(self, df):
        self.data=df.copy()
        if self.data is not None:
            min_match = max(2, len(self.data.columns) // 2)
            header_mask = (self.data == self.data.columns).sum(axis=1) >= min_match
            if header_mask.any():
                self.data = self.data[~header_mask].reset_index(drop=True)
        self.warnings=[]

    def _get_numeric_coerced_df(self):
        """Helper to coerce numeric-majority object columns so audit checks are not blinded by string dtypes."""
        _UNIT_PAT = r'(?i)\s*(gb|mb|kb|tb|pb|yrs?|years?|months?|days?|hours?|hrs?|mins?|sec|seconds?|%|km|miles?|kg|lbs?)\s*$'
        _SYM_PAT = r'[₹\$,/€£¥\s]'
        df_coerced = self.data.copy()
        for col in df_coerced.select_dtypes(include=['object', 'string']).columns:
            sample = df_coerced[col].dropna().astype(str).str.strip()
            if sample.empty or len(sample) < 5:
                continue
            cleaned = sample.str.replace(_UNIT_PAT, '', regex=True).str.replace(_SYM_PAT, '', regex=True).str.replace(',', '.', regex=False).str.strip()
            coerced = pd.to_numeric(cleaned, errors='coerce')
            if coerced.notna().mean() >= 0.70:
                df_coerced[col] = coerced
        return df_coerced

    # 1. Detect ID Columns
    def check_id_columns(self, threshold=0.95):
        try:
            self.id_cols=[]
            for col in self.data.columns:
                unique_ratio=self.data[col].nunique(dropna=True) / len(self.data)
                is_name_id=bool(re.search(r'(id|uuid|key|ssn|index|identifier)$', col.lower()))

                if (unique_ratio >= threshold and is_name_id) or (unique_ratio == 1.0 and self.data[col].dtype == 'object'):
                    self.id_cols.append(col)
                    logger.warning("Potential unique ID column detected: '%s' (uniqueness: %.1f%%)", col, unique_ratio * 100)

            if self.id_cols:
                self.warnings.append({
                    "Check": "ID Columns",
                    "Status": "WARN",
                    "Details": f"Columns {self.id_cols} appear to be unique IDs.",
                    "Fix": "Drop ID columns to prevent overfitting."
                })
            return self.id_cols
        except Exception as e:
            logger.error("Error in check_id_columns: %s", e)
            return []

    # 2. Check Target Leakage
    def check_target_leakage(self, target=None, threshold=0.90):
        try:
            self.leakage_cols=[]
            target_to_use=target or self.target
            if not target_to_use or target_to_use not in self.data.columns:
                return []

            # Use coerced numeric dataframe so string-formatted leakage (e.g. '$1,200', '0.994') is caught
            df_coerced = self._get_numeric_coerced_df()
            num_cols=df_coerced.select_dtypes(include='number').copy()
            if target_to_use not in num_cols.columns:
                try:
                    num_cols[target_to_use]=pd.to_numeric(df_coerced[target_to_use], errors='coerce')
                except Exception:
                    num_cols[target_to_use]=df_coerced[target_to_use].astype('category').cat.codes

            if target_to_use in num_cols.columns:
                corrs=num_cols.corr().abs()[target_to_use]
                for col, val in corrs.items():
                    if col != target_to_use and val >= threshold:
                        self.leakage_cols.append((col, round(val, 4)))
                        logger.error("Suspicious target leakage: feature '%s' has correlation %.4f with target '%s'", col, val, target_to_use)

            if self.leakage_cols:
                self.warnings.append({
                    "Check": "Target Leakage",
                    "Status": "FAIL",
                    "Details": f"Features {self.leakage_cols} have correlation >= {threshold} with target '{target_to_use}'.",
                    "Fix": "Verify if these columns contain future data."
                })
            return self.leakage_cols
        except Exception as e:
            logger.error("Error in check_target_leakage: %s", e)
            return []

    # 3. Check Class Imbalance
    def check_class_imbalance(self, target=None):
        try:
            target_to_use=target or self.target
            if not target_to_use or target_to_use not in self.data.columns:
                return

            target_series = pd.to_numeric(self.data[target_to_use], errors='coerce').dropna()
            if target_series.empty:
                target_series = self.data[target_to_use].dropna()

            counts=target_series.value_counts(normalize=True)
            if len(counts) > 1 and len(counts) <= 20:
                minority_prop=counts.min()
                if minority_prop < 0.10:
                    logger.warning("High class imbalance detected in '%s': minority class is %.2f%%", target_to_use, minority_prop * 100)
                    self.warnings.append({
                        "Check": "Class Imbalance",
                        "Status": "WARN",
                        "Details": f"Minority class proportion is {minority_prop:.2%}.",
                        "Fix": "Use SMOTE, class weights, or stratified cross-validation."
                    })
        except Exception as e:
            logger.error("Error in check_class_imbalance: %s", e)

    # 4. Check High Cardinality
    def check_cardinality(self, threshold=100):
        try:
            self.high_cardinality_cols=[]
            # Use coerced numeric dataframe so numerical columns are not falsely treated as categoricals
            df_coerced = self._get_numeric_coerced_df()
            cat_cols=df_coerced.select_dtypes(include=['object', 'category', 'string']).columns

            pii_cols = []
            date_cols = []
            generic_high_card = []

            for col in cat_cols:
                n_unique=df_coerced[col].nunique()
                if col in self.id_cols:
                    continue

                col_clean = re.sub(r'[^\w]', '', col.lower())
                sample = df_coerced[col].dropna().astype(str).str.strip()

                # Check if it's a datetime column
                is_name_date = bool(re.search(r'(date|time|timestamp|dob)$', col_clean))
                is_parse_date = (
                    pd.to_datetime(sample.iloc[:100], format='mixed', errors='coerce').notna().mean() >= 0.70
                    if not sample.empty else False
                )
                if is_name_date or is_parse_date:
                    date_cols.append((col, n_unique))
                    logger.info("Datetime column identified: '%s' (%d unique timestamps)", col, n_unique)
                    continue

                # Check if it's PII (name, email, phone, etc.)
                is_pii = bool(re.search(r'(name|fullname|firstname|lastname|email|phone|address|ssn)$', col_clean))
                if is_pii:
                    pii_cols.append((col, n_unique))
                    logger.warning("PII column detected: '%s' (%d unique values)", col, n_unique)
                    continue

                if n_unique > threshold:
                    generic_high_card.append((col, n_unique))
                    logger.warning("High cardinality column '%s': %d unique values", col, n_unique)

            if pii_cols:
                self.warnings.append({
                    "Check": "PII Detection",
                    "Status": "WARN",
                    "Details": f"Columns {pii_cols} appear to contain Personally Identifiable Information (PII).",
                    "Fix": "Drop PII columns to protect user privacy and prevent overfitting."
                })

            if date_cols:
                self.warnings.append({
                    "Check": "Datetime Features",
                    "Status": "INFO",
                    "Details": f"Columns {date_cols} are datetime series rather than plain categoricals.",
                    "Fix": "Extract datetime features (year, month, day, dayofweek) via pipeline's data_time parameter."
                })

            if generic_high_card:
                self.warnings.append({
                    "Check": "High Cardinality",
                    "Status": "WARN",
                    "Details": f"Categorical columns {generic_high_card} have > {threshold} unique categories.",
                    "Fix": "Use Target/Frequency Encoding or group rare categories."
                })

            self.high_cardinality_cols = pii_cols + generic_high_card
            return self.high_cardinality_cols
        except Exception as e:
            logger.error("Error in check_cardinality: %s", e)
            return []

    # 5. Check Infinite or Extreme Values
    def check_infinite_values(self):
        try:
            num_cols=self.data.select_dtypes(include='number').columns
            inf_cols=[]

            for col in num_cols:
                inf_count=np.isinf(self.data[col]).sum()
                if inf_count > 0:
                    inf_cols.append((col, int(inf_count)))
                    logger.error("Infinite values detected in '%s': %d occurrences", col, inf_count)

            if inf_cols:
                self.warnings.append({
                    "Check": "Infinite Values",
                    "Status": "FAIL",
                    "Details": f"Columns {inf_cols} contain infinity (+inf or -inf).",
                    "Fix": "Replace infinite values with NaN or bounds."
                })
            return inf_cols
        except Exception as e:
            logger.error("Error in check_infinite_values: %s", e)
            return []

    # ==========================
    # Master Health Audit Runner
    # ==========================
    def run_audit(self, target=None, leakage_threshold=0.90, id_threshold=0.95, cardinality_threshold=100):
        try:
            self.warnings=[]
            logger.info("Running data quality audit on %d rows, %d columns", len(self.data), len(self.data.columns))

            self.check_id_columns(threshold=id_threshold)
            self.check_infinite_values()
            self.check_cardinality(threshold=cardinality_threshold)

            target_to_use=target or self.target
            if target_to_use:
                self.check_target_leakage(target=target_to_use, threshold=leakage_threshold)
                self.check_class_imbalance(target=target_to_use)

            if not self.warnings:
                logger.info("Quality audit completed: all checks passed")
                return pd.DataFrame([{"Check": "All Checks", "Status": "PASS", "Details": "No issues found", "Fix": "None"}])

            report_df=pd.DataFrame(self.warnings)
            logger.info("Quality audit completed: %d warnings/failures flagged", len(self.warnings))
            return report_df
        except Exception as e:
            logger.error("Error in run_audit: %s", e)
            return pd.DataFrame()
