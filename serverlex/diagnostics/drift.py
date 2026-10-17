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
    - Multivariate Drift Summary: Aggregate drift metrics across all incoming production features.
    """

    def __init__(self, reference_data, alpha=0.05, psi_threshold=0.2):
        self.ref_data=reference_data.copy() if isinstance(reference_data, pd.DataFrame) else pd.DataFrame(reference_data)
        self.alpha=alpha
        self.psi_threshold=psi_threshold
        self.drift_results=[]

    @staticmethod
    def calculate_psi(expected, actual, num_bins=10):
        """
        Calculates Population Stability Index (PSI) between reference and production distributions.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate shift
        PSI >= 0.25: Significant population drift
        """
        try:
            expected = np.asarray(expected, dtype=float)
            actual = np.asarray(actual, dtype=float)
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]

            if len(expected) < 10 or len(actual) < 10:
                return 0.0

            # Quantile bins from reference data
            quantiles = np.linspace(0, 100, num_bins + 1)
            bins = np.percentile(expected, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.unique(bins)

            if len(bins) < 3:
                return 0.0

            expected_counts, _ = np.histogram(expected, bins=bins)
            actual_counts, _ = np.histogram(actual, bins=bins)

            # Smooth zero counts
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

            common_cols=[c for c in self.ref_data.columns if c in cur_data.columns]
            logger.info("Running drift detection across %d features (alpha=%.3f, psi_threshold=%.2f)", len(common_cols), self.alpha, self.psi_threshold)

            for col in common_cols:
                # 1. Numerical Drift (KS test, PSI, and Wasserstein distance)
                if pd.api.types.is_numeric_dtype(self.ref_data[col]):
                    ref_vals=self.ref_data[col].dropna()
                    cur_vals=cur_data[col].dropna()

                    if len(ref_vals) > 0 and len(cur_vals) > 0:
                        stat, p_val = ks_2samp(ref_vals, cur_vals)
                        psi = self.calculate_psi(ref_vals, cur_vals)
                        w_dist = float(np.round(wasserstein_distance(ref_vals, cur_vals), 4))

                        is_drift = (p_val < self.alpha) or (psi >= self.psi_threshold)
                        if is_drift:
                            drift_count += 1
                            logger.warning("Data drift detected in '%s' (KS=%.4f, p=%.5f, PSI=%.4f, Wasserstein=%.4f)", col, stat, p_val, psi, w_dist)

                        self.drift_results.append({
                            "Feature": col,
                            "Type": "Numerical",
                            "Statistic": round(stat, 4),
                            "P_Value": round(p_val, 5),
                            "PSI": psi,
                            "Wasserstein": w_dist,
                            "Drift_Detected": is_drift
                        })

                # 2. Categorical Drift (Chi-Square test & Population Stability Index)
                else:
                    ref_counts=self.ref_data[col].astype(str).value_counts()
                    cur_counts=cur_data[col].astype(str).value_counts()

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

                    is_drift = (p_val < self.alpha) or (cat_psi >= self.psi_threshold)
                    if is_drift:
                        drift_count += 1
                        logger.warning("Categorical shift detected in '%s' (Chi2=%.4f, p=%.5f, PSI=%.4f)", col, stat, p_val, cat_psi)

                    self.drift_results.append({
                        "Feature": col,
                        "Type": "Categorical",
                        "Statistic": round(stat, 4),
                        "P_Value": round(p_val, 5),
                        "PSI": cat_psi,
                        "Wasserstein": None,
                        "Drift_Detected": is_drift
                    })

            report_df=pd.DataFrame(self.drift_results)
            logger.info("Drift analysis complete: %d/%d features drifted", drift_count, len(common_cols))
            return report_df
        except Exception as e:
            logger.error("Error in detect_drift: %s", e)
            return pd.DataFrame()
