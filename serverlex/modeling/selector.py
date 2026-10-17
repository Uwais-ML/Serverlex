import logging
import pandas as pd
import numpy as np

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    VarianceThreshold,
    SelectKBest
)

logger=logging.getLogger("FeatureSelector")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class FeatureSelector:
    def __init__(self, k=10, variance_threshold=0.01, corr_threshold=0.90):
        self.k=k
        self.variance_threshold=variance_threshold
        self.corr_threshold=corr_threshold
        self.selected_features=[]
        self.feature_scores={}
        self.is_fitted=False

    def fit(self, X, y):
        try:
            if not isinstance(X, pd.DataFrame):
                X=pd.DataFrame(X)
            if not isinstance(y, pd.Series):
                y=pd.Series(y)

            num_cols=X.select_dtypes(include='number').columns.tolist()
            cur_X=X[num_cols].copy()

            # 1. Prune near-zero variance features
            var_selector=VarianceThreshold(threshold=self.variance_threshold)
            var_selector.fit(cur_X)
            non_zero_cols=cur_X.columns[var_selector.get_support()].tolist()
            cur_X=cur_X[non_zero_cols]

            # 2. Prune collinear features (redundant duplicates)
            corr_matrix=cur_X.corr().abs()
            upper_tri=corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            cols_to_drop=[col for col in upper_tri.columns if any(upper_tri[col] > self.corr_threshold)]
            cur_X=cur_X.drop(columns=cols_to_drop)

            # 3. Mutual Information Feature Importance
            is_class=y.dtype == 'object' or str(y.dtype).startswith('cat') or y.nunique() <= 20
            mi_func=mutual_info_classif if is_class else mutual_info_regression

            mi_scores=mi_func(cur_X, y, random_state=42)
            score_series=pd.Series(mi_scores, index=cur_X.columns).sort_values(ascending=False)
            self.feature_scores=score_series.to_dict()

            # Pick top K features
            k_val=min(self.k, len(score_series))
            self.selected_features=score_series.head(k_val).index.tolist()
            self.is_fitted=True

            logger.info("Feature selection complete: %d features selected from %d initial features", len(self.selected_features), X.shape[1])
            return self
        except Exception as e:
            logger.error("Error in FeatureSelector.fit: %s", e)
            return self

    def transform(self, X):
        try:
            if not self.is_fitted:
                raise ValueError("FeatureSelector is not fitted yet. Call fit() first.")

            if not isinstance(X, pd.DataFrame):
                X=pd.DataFrame(X)

            # Align to selected features (fill missing with 0)
            return X.reindex(columns=self.selected_features, fill_value=0)
        except Exception as e:
            logger.error("Error in FeatureSelector.transform: %s", e)
            return X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_ranked_scores(self):
        """Returns DataFrame of features ranked by Mutual Information."""
        return pd.DataFrame(list(self.feature_scores.items()), columns=["Feature", "Mutual_Information_Score"])

    @staticmethod
    def prune_collinear(X, threshold=0.85, y=None):
        """
        Standalone collinearity pruning utility:
        - Finds pairs of features with Pearson |r| >= threshold.
        - If target 'y' is provided, keeps the feature with higher predictive correlation with y.
        - If 'y' is None, drops the second feature in the pair.

        Returns:
            pruned_X (pd.DataFrame): DataFrame with collinear duplicates removed.
            dropped_cols (list): Names of features dropped due to collinearity.
        """
        try:
            if not isinstance(X, pd.DataFrame):
                df = pd.DataFrame(X)
            else:
                df = X.copy()

            num_cols = df.select_dtypes(include='number').columns.tolist()
            if len(num_cols) < 2:
                return df, []

            corr_matrix = df[num_cols].corr().abs()
            dropped_cols = set()

            # Pre-compute correlation with target y if provided
            y_scores = {}
            if y is not None:
                y_series = pd.to_numeric(y, errors='coerce') if not isinstance(y, pd.Series) else y
                for col in num_cols:
                    try:
                        valid_mask = df[col].notna() & y_series.notna()
                        if valid_mask.sum() > 5:
                            corr_val = float(np.corrcoef(df[col][valid_mask], y_series[valid_mask])[0, 1])
                            y_scores[col] = abs(corr_val) if not np.isnan(corr_val) else 0.0
                        else:
                            y_scores[col] = 0.0
                    except Exception:
                        y_scores[col] = 0.0

            for i in range(len(num_cols)):
                col_i = num_cols[i]
                if col_i in dropped_cols:
                    continue
                for j in range(i + 1, len(num_cols)):
                    col_j = num_cols[j]
                    if col_j in dropped_cols:
                        continue
                    r_val = corr_matrix.loc[col_i, col_j]
                    if not np.isnan(r_val) and r_val >= threshold:
                        if y_scores:
                            if y_scores.get(col_i, 0) >= y_scores.get(col_j, 0):
                                dropped_cols.add(col_j)
                            else:
                                dropped_cols.add(col_i)
                                break
                        else:
                            dropped_cols.add(col_j)

            dropped_list = sorted(list(dropped_cols))
            pruned_df = df.drop(columns=dropped_list)
            logger.info("prune_collinear: Dropped %d collinear feature(s) (|r| >= %.2f): %s", len(dropped_list), threshold, dropped_list)
            return pruned_df, dropped_list
        except Exception as e:
            logger.error("Error in prune_collinear: %s", e)
            return X, []

