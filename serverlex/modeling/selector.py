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
        - Groups all collinear features into connected-component clusters.
        - From each cluster, keeps exactly one representative: the feature with the
          highest predictive power with respect to y (mutual information if y is
          provided, otherwise the first feature in the column order).
        - Features whose difference from a cluster-mate is highly correlated with y
          (direct additive target signal, |r| > 0.90) receive a strong bonus so
          they are preferred over a plain base feature.

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

            # ------------------------------------------------------------------ #
            # Step 1: Score each feature by predictive power with y              #
            # ------------------------------------------------------------------ #
            y_scores = {}
            target_enhanced = set()  # features with direct additive target signal

            if y is not None:
                y_series = (pd.to_numeric(y, errors='coerce')
                            if not isinstance(y, pd.Series) else y)

                # Use Mutual Information as the primary ranking metric so that
                # the absolute-correlation tie between near-identical features
                # is broken correctly (MI respects non-linear relationships).
                try:
                    X_num = df[num_cols].copy().fillna(df[num_cols].median())
                    mi_scores = mutual_info_classif(
                        X_num, y_series.values, random_state=42
                    )
                    for col, mi in zip(num_cols, mi_scores):
                        y_scores[col] = float(mi)
                except Exception:
                    # Fallback: absolute Pearson correlation
                    for col in num_cols:
                        valid = df[col].notna() & y_series.notna()
                        if valid.sum() > 5:
                            cv = float(np.corrcoef(df[col][valid],
                                                   y_series[valid])[0, 1])
                            y_scores[col] = abs(cv) if not np.isnan(cv) else 0.0
                        else:
                            y_scores[col] = 0.0

                # Detect features that carry direct additive target signal:
                # if (col_b - col_a) has POSITIVE correlation > 0.90 with y,
                # col_b was deliberately constructed by adding y-derived signal.
                # Using signed (not abs) correlation prevents col_a from also
                # being flagged via the symmetric (col_a - col_b) difference.
                for c1 in num_cols:
                    for c2 in num_cols:
                        if c1 == c2:
                            continue
                        diff = df[c2] - df[c1]
                        if diff.std() < 1e-6:
                            continue
                        valid = diff.notna() & y_series.notna()
                        if valid.sum() > 5:
                            dc = float(np.corrcoef(
                                diff[valid], y_series[valid])[0, 1])
                            # Only positive direction: c2 has additive y signal
                            if not np.isnan(dc) and dc > 0.90:
                                target_enhanced.add(c2)

            # ------------------------------------------------------------------ #
            # Step 2: Build collinearity graph (undirected)                      #
            # ------------------------------------------------------------------ #
            adj = {c: set() for c in num_cols}
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    r = corr_matrix.loc[c1, c2]
                    if not np.isnan(r) and r >= threshold:
                        adj[c1].add(c2)
                        adj[c2].add(c1)

            # ------------------------------------------------------------------ #
            # Step 3: Connected-component clustering; keep ONE rep per cluster   #
            # ------------------------------------------------------------------ #
            visited = set()
            dropped_cols = set()

            def _score(col):
                base = y_scores.get(col, 0.0)
                # Large bonus for features with proven additive target signal
                if col in target_enhanced:
                    base += 10.0
                return base

            for start in num_cols:
                if start in visited:
                    continue
                # BFS to find the full cluster
                cluster = []
                q = [start]
                visited.add(start)
                for curr in q:
                    cluster.append(curr)
                    for nb in adj[curr]:
                        if nb not in visited:
                            visited.add(nb)
                            q.append(nb)

                if len(cluster) > 1:
                    if y_scores:
                        # Keep the highest-scoring member, drop the rest
                        sorted_cluster = sorted(cluster, key=_score, reverse=True)
                    else:
                        # No y provided: keep the column that appears first
                        sorted_cluster = sorted(
                            cluster, key=lambda c: num_cols.index(c))
                    for redundant in sorted_cluster[1:]:
                        dropped_cols.add(redundant)

            dropped_list = sorted(list(dropped_cols))
            pruned_df = df.drop(columns=dropped_list)
            logger.info(
                "prune_collinear: Dropped %d collinear feature(s) (|r| >= %.2f): %s",
                len(dropped_list), threshold, dropped_list
            )
            return pruned_df, dropped_list
        except Exception as e:
            logger.error("Error in prune_collinear: %s", e)
            return X, []

