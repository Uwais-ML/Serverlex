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
    def __init__(self, df=None):
        self.data=df.copy() if df is not None else None
        self.warnings=[]
        self.id_cols=[]
        self.leakage_cols=[]
        self.high_cardinality_cols=[]

    def set_data(self, df):
        self.data=df.copy()
        self.warnings=[]

    # 1. Detect ID Columns
    def check_id_columns(self):
        try:
            self.id_cols=[]
            for col in self.data.columns:
                unique_ratio=self.data[col].nunique(dropna=True) / len(self.data)
                is_name_id=bool(re.search(r'(id|uuid|key|ssn|index|identifier)$', col.lower()))

                if (unique_ratio > 0.95 and is_name_id) or (unique_ratio == 1.0 and self.data[col].dtype == 'object'):
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

    # 2. Check Target Leakage
    def check_target_leakage(self, target, threshold=0.95):
        try:
            self.leakage_cols=[]
            if target not in self.data.columns:
                return

            num_cols=self.data.select_dtypes(include='number')
            if target in num_cols.columns:
                corrs=num_cols.corr().abs()[target]
                for col, val in corrs.items():
                    if col != target and val >= threshold:
                        self.leakage_cols.append((col, round(val, 4)))
                        logger.error("Suspicious target leakage: feature '%s' has correlation %.4f with target '%s'", col, val, target)

            if self.leakage_cols:
                self.warnings.append({
                    "Check": "Target Leakage",
                    "Status": "FAIL",
                    "Details": f"Features {self.leakage_cols} have correlation >= {threshold} with target '{target}'.",
                    "Fix": "Verify if these columns contain future data."
                })
            return self.leakage_cols
        except Exception as e:
            logger.error("Error in check_target_leakage: %s", e)

    # 3. Check Class Imbalance
    def check_class_imbalance(self, target):
        try:
            if target not in self.data.columns:
                return

            counts=self.data[target].value_counts(normalize=True)
            if len(counts) <= 20: # Classification
                max_prop=counts.max()
                min_prop=counts.min()
                imbalance_ratio=round(max_prop / min_prop, 2)

                if max_prop >= 0.70:
                    status="WARN" if max_prop < 0.90 else "FAIL"
                    logger.warning("Class imbalance in '%s': dominant class is %.1f%% (ratio %.2f:1)", target, max_prop * 100, imbalance_ratio)
                    self.warnings.append({
                        "Check": "Class Imbalance",
                        "Status": status,
                        "Details": f"Dominant class represents {max_prop:.1%} (ratio: {imbalance_ratio}:1).",
                        "Fix": "Use class_weight='balanced' or resampling."
                    })
        except Exception as e:
            logger.error("Error in check_class_imbalance: %s", e)

    # 4. Check High-Cardinality Categoricals
    def check_cardinality(self, threshold=30):
        try:
            self.high_cardinality_cols=[]
            cat_cols=self.data.select_dtypes(include=['object', 'category']).columns

            for col in cat_cols:
                n_unique=self.data[col].nunique()
                if n_unique > threshold:
                    self.high_cardinality_cols.append((col, n_unique))
                    logger.warning("High cardinality in category '%s': %d unique values", col, n_unique)

            if self.high_cardinality_cols:
                self.warnings.append({
                    "Check": "High Cardinality",
                    "Status": "WARN",
                    "Details": f"Columns {self.high_cardinality_cols} exceed {threshold} unique values.",
                    "Fix": "Use label encoding or group rare categories."
                })
        except Exception as e:
            logger.error("Error in check_cardinality: %s", e)

    # 5. Check Infinite Values
    def check_infinite_values(self):
        try:
            num_cols=self.data.select_dtypes(include='number').columns
            inf_cols=[]
            for col in num_cols:
                if np.isinf(self.data[col]).any():
                    inf_cols.append(col)
                    logger.error("Infinite values detected in column '%s'", col)

            if inf_cols:
                self.warnings.append({
                    "Check": "Infinite Values",
                    "Status": "FAIL",
                    "Details": f"Columns {inf_cols} contain inf / -inf.",
                    "Fix": "Replace inf values with NaN or bounds."
                })
        except Exception as e:
            logger.error("Error in check_infinite_values: %s", e)

    # ==========================
    # Master Health Audit Runner
    # ==========================
    def run_audit(self, target=None):
        try:
            self.warnings=[]
            logger.info("Running data quality audit on %d rows, %d columns", len(self.data), len(self.data.columns))

            self.check_id_columns()
            self.check_infinite_values()
            self.check_cardinality()

            if target:
                self.check_target_leakage(target)
                self.check_class_imbalance(target)

            if not self.warnings:
                logger.info("Quality audit completed: all checks passed")
                return pd.DataFrame([{"Check": "All Checks", "Status": "PASS", "Details": "No issues found", "Fix": "None"}])

            report_df=pd.DataFrame(self.warnings)
            logger.info("Quality audit completed: %d warnings/failures flagged", len(self.warnings))
            return report_df
        except Exception as e:
            logger.error("Error in run_audit: %s", e)
