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
