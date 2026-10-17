import logging
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance

logger=logging.getLogger("DriftDetector")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class DriftDetector:
    """
    Enterprise Data & Population Drift Monitoring Engine:
    - Univariate Statistical Testing: 2-sample Kolmogorov-Smirnov & Chi-Square Contingency.
    - Population Stability Index (PSI): Industry-standard binned population shift metric.
    - Wasserstein Distance: Earth Mover's Distance for continuous geometric distribution shift.
    - Missing Feature Audit: Explicitly alerts on columns dropped in production instead of silently skipping them.
    - Balanced Triggering: Avoids false alarms from hypersensitive p-values by combining p-value with PSI thresholds.
    """

    def __init__(self, reference_data, alpha=0.01, psi_threshold=0.2):
        self.ref_data=reference_data.copy() if isinstance(reference_data, pd.DataFrame) else pd.DataFrame(reference_data)
        self.alpha=alpha
        self.psi_threshold=psi_threshold
        self.drift_results=[]

    @staticmethod
    def calculate_psi(expected, actual, num_bins=10):
        try:
            expected = np.asarray(expected, dtype=float)
            actual = np.asarray(actual, dtype=float)
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]

            if len(expected) < 10 or len(actual) < 10:
                return 0.0

            quantiles = np.linspace(0, 100, num_bins + 1)
            bins = np.percentile(expected, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.unique(bins)

            if len(bins) < 3:
                return 0.0

            expected_counts, _ = np.histogram(expected, bins=bins)
            actual_counts, _ = np.histogram(actual, bins=bins)

            expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts) / len(expected)
            actual_pct = np.where(actual_counts == 0, 0.0001, actual_counts) / len(actual)

            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            return float(np.round(psi, 4))
        except Exception:
            return 0.0

    def detect_drift(self, current_data):
        try:
            cur_data=current_data.copy() if isinstance(current_data, pd.DataFrame) else pd.DataFrame(current_data)
            self.drift_results=[]
            drift_count=0

            ref_cols = list(self.ref_data.columns)
            cur_cols = list(cur_data.columns)

            # 1. Audit Missing Columns
            missing_in_prod = [c for c in ref_cols if c not in cur_cols]
            for col in missing_in_prod:
                drift_count += 1
                logger.error("Missing feature in production data: '%s'", col)
                self.drift_results.append({
                    "Feature": col,
                    "Type": "Missing Feature",
                    "Statistic": None,
                    "P_Value": None,
                    "PSI": 1.0,
                    "Wasserstein": None,
                    "Drift_Detected": True,
                    "Severity": "Critical (Missing)"
                })

            common_cols = [c for c in ref_cols if c in cur_cols]
            logger.info("Running drift detection across %d features (alpha=%.3f, psi_threshold=%.2f)", len(ref_cols), self.alpha, self.psi_threshold)

            for col in common_cols:
                ref_series = self.ref_data[col].dropna()
                cur_series = cur_data[col].dropna()

                if len(ref_series) == 0 or len(cur_series) == 0:
                    continue

                # 2. Check Constant Shift (was constant in reference, altered in current)
                if ref_series.nunique() <= 1:
                    ref_val = ref_series.iloc[0] if len(ref_series) > 0 else None
                    is_shifted = (cur_series.nunique() > 1) or (cur_series.nunique() == 1 and cur_series.iloc[0] != ref_val)
                    if is_shifted:
                        drift_count += 1
                        logger.warning("Constant feature shift detected in '%s' (ref=%s, cur_unique=%s)", col, ref_val, cur_series.unique())
                        self.drift_results.append({
                            "Feature": col,
                            "Type": "Constant Shift",
                            "Statistic": None,
                            "P_Value": 0.0,
                            "PSI": 1.0,
                            "Wasserstein": None,
                            "Drift_Detected": True,
                            "Severity": "High"
                        })
                        continue

                # 3. Numerical Drift
                if pd.api.types.is_numeric_dtype(self.ref_data[col]):
                    ref_vals = ref_series
                    cur_vals = cur_series

                    stat, p_val = ks_2samp(ref_vals, cur_vals)
                    psi = self.calculate_psi(ref_vals, cur_vals)
                    w_dist = float(np.round(wasserstein_distance(ref_vals, cur_vals), 4))

                    # Guard against small sample noise: Reliable drift analysis requires sample size >= 50
                    if len(cur_vals) < 50:
                        is_drift = False
                        severity = "Insufficient Samples (< 50)"
                    else:
                        # Correct for finite-sample PSI expectation floor E[PSI] approx (B-1)*(1/n_cur + 1/n_ref)
                        psi_floor = 9.0 * (1.0 / len(cur_vals) + 1.0 / len(ref_vals))
                        effective_threshold = max(self.psi_threshold, 2.0 * psi_floor)

                        # Require BOTH statistical significance (p < alpha) AND practical significance (PSI >= threshold)
                        is_drift = (psi >= effective_threshold) and (p_val < self.alpha)

                        if not is_drift:
                            severity = "None"
                        elif psi >= max(0.25, 2.5 * psi_floor):
                            severity = "High"
                        else:
                            severity = "Moderate"

                    if is_drift:
                        drift_count += 1
                        logger.warning("Data drift detected in '%s' (KS=%.4f, p=%.5f, PSI=%.4f, Wasserstein=%.4f, Severity=%s)", col, stat, p_val, psi, w_dist, severity)

                    self.drift_results.append({
                        "Feature": col,
                        "Type": "Numerical",
                        "Statistic": round(stat, 4),
                        "P_Value": round(p_val, 5),
                        "PSI": psi,
                        "Wasserstein": w_dist,
                        "Drift_Detected": is_drift,
                        "Severity": severity
                    })

                # 4. Categorical Drift
                else:
                    ref_counts=self.ref_data[col].astype(str).value_counts()
                    cur_counts=cur_data[col].astype(str).value_counts()

                    # Guard against small batches: Chi-square and PSI are unstable for n < 50
                    if len(cur_series) < 50:
                        self.drift_results.append({
                            "Feature": col,
                            "Type": "Categorical",
                            "Statistic": None,
                            "P_Value": 1.0,
                            "PSI": 0.0,
                            "Wasserstein": None,
                            "Drift_Detected": False,
                            "Severity": "Insufficient Samples (< 50)"
                        })
                        continue

                    all_cats=list(set(ref_counts.index).union(set(cur_counts.index)))
                    ref_aligned=np.array([ref_counts.get(cat, 0) for cat in all_cats], dtype=float)
                    cur_aligned=np.array([cur_counts.get(cat, 0) for cat in all_cats], dtype=float)

                    stat, p_val = 0.0, 1.0
                    contingency_table=np.array([ref_aligned, cur_aligned])
                    if contingency_table.sum() > 0:
                        try:
                            stat, p_val, _, _ = chi2_contingency(contingency_table)
                        except Exception:
                            stat, p_val = 0.0, 1.0

                    ref_pct = np.where(ref_aligned == 0, 0.0001, ref_aligned) / ref_aligned.sum()
                    cur_pct = np.where(cur_aligned == 0, 0.0001, cur_aligned) / cur_aligned.sum()
                    cat_psi = float(np.round(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)), 4))

                    cat_bins = max(2, len(all_cats))
                    cat_psi_floor = (cat_bins - 1) * (1.0 / len(cur_series) + 1.0 / len(ref_series))
                    effective_cat_threshold = max(self.psi_threshold, 2.0 * cat_psi_floor)

                    # Require BOTH statistical significance and practical shift
                    is_drift = (cat_psi >= effective_cat_threshold) and (p_val < self.alpha)

                    if not is_drift:
                        severity = "None"
                    elif cat_psi >= max(0.25, 2.5 * cat_psi_floor):
                        severity = "High"
                    else:
                        severity = "Moderate"

                    if is_drift:
                        drift_count += 1
                        logger.warning("Categorical shift detected in '%s' (Chi2=%.4f, p=%.5f, PSI=%.4f, Severity=%s)", col, stat, p_val, cat_psi, severity)

                    self.drift_results.append({
                        "Feature": col,
                        "Type": "Categorical",
                        "Statistic": round(stat, 4),
                        "P_Value": round(p_val, 5),
                        "PSI": cat_psi,
                        "Wasserstein": None,
                        "Drift_Detected": is_drift,
                        "Severity": severity
                    })

            report_df=pd.DataFrame(self.drift_results)
            logger.info("Drift analysis complete: %d/%d features drifted or missing", drift_count, len(ref_cols))
            return report_df
        except Exception as e:
            logger.error("Error in detect_drift: %s", e)
            raise
