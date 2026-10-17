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
        self.pii_cols=[]
        self.collinear_pairs=[]

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

            self.pii_cols = [c[0] for c in pii_cols]
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

    # 6. Check Collinearity
    def check_collinearity(self, threshold=0.85):
        try:
            self.collinear_pairs = []
            df_coerced = self._get_numeric_coerced_df()
            num_cols = df_coerced.select_dtypes(include='number').columns.tolist()
            if self.target and self.target in num_cols:
                num_cols.remove(self.target)
            if len(num_cols) < 2:
                return []

            corr_matrix = df_coerced[num_cols].corr().abs()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    col_a, col_b = num_cols[i], num_cols[j]
                    r_val = corr_matrix.loc[col_a, col_b]
                    if not np.isnan(r_val) and r_val >= threshold:
                        self.collinear_pairs.append((col_a, col_b, round(float(r_val), 4)))
                        logger.warning("Collinear features detected: '%s' and '%s' (|r| = %.4f >= %.2f)", col_a, col_b, r_val, threshold)

            if self.collinear_pairs:
                self.warnings.append({
                    "Check": "Collinearity",
                    "Status": "WARN",
                    "Details": f"Feature pairs {self.collinear_pairs} have high collinearity (|r| >= {threshold}).",
                    "Fix": "Prune redundant collinear features using FeatureSelector.prune_collinear()."
                })
            return self.collinear_pairs
        except Exception as e:
            logger.error("Error in check_collinearity: %s", e)
            return []

    # ==========================
    # Master Health Audit Runner
    # ==========================
    def run_audit(self, target=None, leakage_threshold=0.90, id_threshold=0.95, cardinality_threshold=100, collinearity_threshold=0.90):
        try:
            self.warnings=[]
            logger.info("Running data quality audit on %d rows, %d columns", len(self.data), len(self.data.columns))

            self.check_id_columns(threshold=id_threshold)
            self.check_infinite_values()
            self.check_cardinality(threshold=cardinality_threshold)
            if collinearity_threshold:
                self.check_collinearity(threshold=collinearity_threshold)

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

    # ==========================
    # Auto-Remediation Engine
    # ==========================
    def auto_remediate(self, target=None, drop_ids=True, drop_pii=True, drop_leakage=True, drop_collinear=False, leakage_threshold=0.90, id_threshold=0.95, collinearity_threshold=0.90):
        """
        Automatically cleans and prunes raw dataset according to audit checks:
        - Drops confirmed ID columns
        - Drops PII columns
        - Drops target leakage candidates (|r| >= leakage_threshold)
        - Optionally drops redundant collinear duplicates

        Returns:
            clean_df (pd.DataFrame): Sanitized dataframe ready for pipeline modeling.
            summary (dict): Audit trail detailing every dropped column and category.
        """
        target_to_use = target or self.target
        self.run_audit(target=target_to_use, leakage_threshold=leakage_threshold, id_threshold=id_threshold, collinearity_threshold=collinearity_threshold)

        df_clean = self.data.copy()
        dropped_cols = {}

        if drop_ids and self.id_cols:
            ids_to_drop = [c for c in self.id_cols if c in df_clean.columns]
            if ids_to_drop:
                df_clean = df_clean.drop(columns=ids_to_drop)
                dropped_cols["id_columns"] = ids_to_drop
                logger.info("AutoRemediate: Dropped %d ID column(s): %s", len(ids_to_drop), ids_to_drop)

        if drop_pii and hasattr(self, 'pii_cols') and self.pii_cols:
            pii_to_drop = [c for c in self.pii_cols if c in df_clean.columns and c != target_to_use]
            if pii_to_drop:
                df_clean = df_clean.drop(columns=pii_to_drop)
                dropped_cols["pii_columns"] = pii_to_drop
                logger.info("AutoRemediate: Dropped %d PII column(s): %s", len(pii_to_drop), pii_to_drop)

        if drop_leakage and self.leakage_cols:
            leakage_to_drop = [col for col, _ in self.leakage_cols if col in df_clean.columns and col != target_to_use]
            if leakage_to_drop:
                df_clean = df_clean.drop(columns=leakage_to_drop)
                dropped_cols["leakage_columns"] = leakage_to_drop
                logger.info("AutoRemediate: Dropped %d target leakage column(s): %s", len(leakage_to_drop), leakage_to_drop)

        if drop_collinear and hasattr(self, 'collinear_pairs') and self.collinear_pairs:
            collinear_to_drop = []
            for col_a, col_b, _ in self.collinear_pairs:
                if col_b in df_clean.columns and col_b != target_to_use and col_b not in collinear_to_drop:
                    collinear_to_drop.append(col_b)
            if collinear_to_drop:
                df_clean = df_clean.drop(columns=collinear_to_drop)
                dropped_cols["collinear_columns"] = collinear_to_drop
                logger.info("AutoRemediate: Dropped %d collinear column(s): %s", len(collinear_to_drop), collinear_to_drop)

        remediation_summary = {
            "initial_columns": len(self.data.columns),
            "remaining_columns": len(df_clean.columns),
            "dropped_columns": dropped_cols,
            "total_dropped": sum(len(v) for v in dropped_cols.values())
        }
        return df_clean, remediation_summary

    # ==========================
    # Standalone HTML Report
    # ==========================
    def to_html_report(self, filepath=None, title="Serverlex Data Health & Quality Audit"):
        """
        Generates a standalone, dependency-free interactive HTML audit report with health score,
        summary metrics cards, color-coded checks, and column profiles.
        """
        if not self.warnings:
            self.run_audit()

        n_rows = len(self.data) if self.data is not None else 0
        n_cols = len(self.data.columns) if self.data is not None else 0
        missing_count = int(self.data.isna().sum().sum()) if self.data is not None else 0
        total_cells = max(1, n_rows * n_cols)
        missing_pct = round((missing_count / total_cells) * 100, 2)
        duplicate_rows = int(self.data.duplicated().sum()) if self.data is not None else 0

        # Calculate Health Score (100 base, -20 per FAIL, -10 per WARN)
        fails = sum(1 for w in self.warnings if w.get("Status") == "FAIL")
        warns = sum(1 for w in self.warnings if w.get("Status") == "WARN")
        passes = max(0, 6 - fails - warns)
        score = max(0, min(100, 100 - (fails * 20) - (warns * 10)))
        if score >= 90:
            grade, grade_color = "A+", "#10b981"
        elif score >= 80:
            grade, grade_color = "A", "#059669"
        elif score >= 70:
            grade, grade_color = "B", "#d97706"
        elif score >= 60:
            grade, grade_color = "C", "#f59e0b"
        else:
            grade, grade_color = "F", "#ef4444"

        # Rows for Audit Checks table
        check_rows = ""
        for w in self.warnings:
            st = w.get("Status", "INFO")
            badge_class = "badge-fail" if st == "FAIL" else ("badge-warn" if st == "WARN" else "badge-pass")
            check_rows += f"""
            <tr>
                <td style="font-weight:600;">{w.get('Check')}</td>
                <td><span class="badge {badge_class}">{st}</span></td>
                <td>{w.get('Details')}</td>
                <td style="color:#94a3b8; font-style:italic;">{w.get('Fix')}</td>
            </tr>"""

        if not check_rows:
            check_rows = """
            <tr>
                <td style="font-weight:600;">All Checks</td>
                <td><span class="badge badge-pass">PASS</span></td>
                <td>All pre-flight data quality checks passed cleanly with zero critical defects.</td>
                <td style="color:#94a3b8;">None</td>
            </tr>"""

        # Rows for Column Profiles table
        col_rows = ""
        id_set = set(getattr(self, 'id_cols', []))
        pii_set = set(getattr(self, 'pii_cols', []))
        leakage_set = {col for col, _ in getattr(self, 'leakage_cols', [])}

        for col in (self.data.columns if self.data is not None else []):
            dtype = str(self.data[col].dtype)
            null_cnt = int(self.data[col].isna().sum())
            null_pct_col = round((null_cnt / max(1, n_rows)) * 100, 1)
            nunique = int(self.data[col].nunique(dropna=True))

            tags = []
            if self.target and col == self.target:
                tags.append('<span class="tag tag-target">Target</span>')
            if col in id_set:
                tags.append('<span class="tag tag-id">Unique ID</span>')
            if col in pii_set:
                tags.append('<span class="tag tag-pii">PII</span>')
            if col in leakage_set:
                tags.append('<span class="tag tag-leakage">Leakage</span>')
            if not tags:
                tags.append(f'<span class="tag tag-default">{dtype}</span>')

            col_rows += f"""
            <tr>
                <td style="font-weight:600; font-family:monospace;">{col}</td>
                <td style="color:#cbd5e1;">{dtype}</td>
                <td>{null_cnt} ({null_pct_col}%)</td>
                <td>{nunique}</td>
                <td>{' '.join(tags)}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background:#0f172a; color:#f8fafc; padding:32px 24px; }}
        .container {{ max-width:1200px; margin:0 auto; }}
        .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:28px; border-bottom:1px solid #334155; padding-bottom:20px; }}
        .header h1 {{ font-size:24px; font-weight:700; color:#38bdf8; display:flex; align-items:center; gap:10px; }}
        .score-box {{ background:#1e293b; border:1px solid #334155; padding:12px 20px; border-radius:12px; display:flex; align-items:center; gap:16px; }}
        .score-val {{ font-size:32px; font-weight:800; color:{grade_color}; }}
        .score-label {{ font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; }}
        .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:16px; margin-bottom:32px; }}
        .card {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:18px; }}
        .card-num {{ font-size:26px; font-weight:700; color:#f1f5f9; margin-top:6px; }}
        .card-title {{ font-size:13px; color:#94a3b8; text-transform:uppercase; font-weight:600; letter-spacing:0.5px; }}
        .section-title {{ font-size:18px; font-weight:700; color:#e2e8f0; margin-bottom:14px; margin-top:24px; display:flex; align-items:center; gap:8px; }}
        table {{ width:100%; border-collapse:collapse; background:#1e293b; border:1px solid #334155; border-radius:12px; overflow:hidden; margin-bottom:32px; }}
        th, td {{ padding:12px 16px; text-align:left; font-size:14px; border-bottom:1px solid #334155; }}
        th {{ background:#0f172a; color:#94a3b8; text-transform:uppercase; font-size:11px; font-weight:700; letter-spacing:0.5px; }}
        tr:last-child td {{ border-bottom:none; }}
        tr:hover td {{ background:#243248; }}
        .badge {{ padding:4px 10px; border-radius:9999px; font-size:11px; font-weight:700; text-transform:uppercase; display:inline-block; }}
        .badge-pass {{ background:rgba(16,185,129,0.15); color:#34d399; border:1px solid #10b981; }}
        .badge-warn {{ background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid #f59e0b; }}
        .badge-fail {{ background:rgba(239,68,68,0.15); color:#f87171; border:1px solid #ef4444; }}
        .tag {{ padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; display:inline-block; }}
        .tag-target {{ background:#3b82f6; color:#fff; }}
        .tag-id {{ background:#ef4444; color:#fff; }}
        .tag-pii {{ background:#f59e0b; color:#000; }}
        .tag-leakage {{ background:#dc2626; color:#fff; }}
        .tag-default {{ background:#334155; color:#94a3b8; }}
        .footer {{ text-align:center; margin-top:40px; font-size:12px; color:#64748b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Serverlex ⚡ Data Health Audit</h1>
                <p style="color:#94a3b8; font-size:13px; margin-top:4px;">Pre-Flight Quality, Hygiene, and Leakage Assessment</p>
            </div>
            <div class="score-box">
                <div class="score-val">{score}/100</div>
                <div>
                    <div class="score-label">Health Score ({grade})</div>
                    <div style="font-size:12px; color:#cbd5e1;">{fails} Critical &bull; {warns} Warnings</div>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">Dataset Rows</div>
                <div class="card-num">{n_rows:,}</div>
            </div>
            <div class="card">
                <div class="card-title">Columns</div>
                <div class="card-num">{n_cols}</div>
            </div>
            <div class="card">
                <div class="card-title">Missing Cells</div>
                <div class="card-num">{missing_pct}% <span style="font-size:13px; color:#94a3b8;">({missing_count})</span></div>
            </div>
            <div class="card">
                <div class="card-title">Duplicate Rows</div>
                <div class="card-num">{duplicate_rows}</div>
            </div>
        </div>

        <div class="section-title">🛡️ Diagnostic Quality & Security Checks</div>
        <table>
            <thead>
                <tr>
                    <th style="width:20%;">Check</th>
                    <th style="width:12%;">Status</th>
                    <th style="width:40%;">Details</th>
                    <th style="width:28%;">Remediation</th>
                </tr>
            </thead>
            <tbody>
                {check_rows}
            </tbody>
        </table>

        <div class="section-title">📋 Column-Level Feature Directory</div>
        <table>
            <thead>
                <tr>
                    <th>Column Name</th>
                    <th>Inferred Dtype</th>
                    <th>Missing Values</th>
                    <th>Unique Values</th>
                    <th>Classification / Tag</th>
                </tr>
            </thead>
            <tbody>
                {col_rows}
            </tbody>
        </table>

        <div class="footer">
            Generated by <strong>Serverlex</strong> &bull; Production-Safe ML Preprocessing, Diagnostics, & Serving
        </div>
    </div>
</body>
</html>"""
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info("Saved DataGuard HTML health report to '%s'", filepath)
        return html

