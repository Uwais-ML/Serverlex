import logging
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

logger=logging.getLogger("DriftDetector")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class DriftDetector:
    def __init__(self, reference_data, alpha=0.05):
        """
        reference_data: Historical baseline (training data)
        alpha: Significance threshold for statistical hypothesis tests (default: 0.05)
        """
        self.ref_data=reference_data.copy() if isinstance(reference_data, pd.DataFrame) else pd.DataFrame(reference_data)
        self.alpha=alpha
        self.drift_results=[]

    def detect_drift(self, current_data):
        """
        Compares incoming production data against reference data:
        - Numerical features: Two-Sample Kolmogorov-Smirnov test (ks_2samp)
        - Categorical features: Chi-Square Contingency test
        """
        try:
            cur_data=current_data.copy() if isinstance(current_data, pd.DataFrame) else pd.DataFrame(current_data)
            self.drift_results=[]
            drift_count=0

            common_cols=[c for c in self.ref_data.columns if c in cur_data.columns]
            logger.info("Running drift detection across %d features (alpha=%.3f)", len(common_cols), self.alpha)

            for col in common_cols:
                # 1. Numerical Drift via Kolmogorov-Smirnov test
                if pd.api.types.is_numeric_dtype(self.ref_data[col]):
                    ref_vals=self.ref_data[col].dropna()
                    cur_vals=cur_data[col].dropna()

                    if len(ref_vals) > 0 and len(cur_vals) > 0:
                        stat, p_val = ks_2samp(ref_vals, cur_vals)
                        is_drift = p_val < self.alpha
                        if is_drift:
                            drift_count += 1
                            logger.warning("Data drift detected in '%s' (KS=%.4f, p-value=%.5f)", col, stat, p_val)

                        self.drift_results.append({
                            "Feature": col,
                            "Type": "Numerical",
                            "Statistic": round(stat, 4),
                            "P_Value": round(p_val, 5),
                            "Drift_Detected": is_drift
                        })

                # 2. Categorical Drift via Chi-Square test
                else:
                    ref_counts=self.ref_data[col].astype(str).value_counts()
                    cur_counts=cur_data[col].astype(str).value_counts()

                    # Align categories
                    all_cats=list(set(ref_counts.index).union(set(cur_counts.index)))
                    ref_aligned=[ref_counts.get(cat, 0) for cat in all_cats]
                    cur_aligned=[cur_counts.get(cat, 0) for cat in all_cats]

                    contingency_table=np.array([ref_aligned, cur_aligned])
                    if contingency_table.sum() > 0:
                        stat, p_val, _, _ = chi2_contingency(contingency_table)
                        is_drift = p_val < self.alpha
                        if is_drift:
                            drift_count += 1
                            logger.warning("Categorical shift detected in '%s' (Chi2=%.4f, p-value=%.5f)", col, stat, p_val)

                        self.drift_results.append({
                            "Feature": col,
                            "Type": "Categorical",
                            "Statistic": round(stat, 4),
                            "P_Value": round(p_val, 5),
                            "Drift_Detected": is_drift
                        })

            report_df=pd.DataFrame(self.drift_results)
            logger.info("Drift analysis complete: %d/%d features drifted", drift_count, len(common_cols))
            return report_df
        except Exception as e:
            logger.error("Error in detect_drift: %s", e)
            return pd.DataFrame()
