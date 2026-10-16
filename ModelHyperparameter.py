import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

logger=logging.getLogger("HyperparameterTuner")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class ModelHyperparameter:
    @staticmethod
    def get_default_param_grid(model_name: str, task: str = "classification"):
        name = model_name.lower().replace(" ", "_")

        if "forest" in name or name == "rf":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            }
        elif "tree" in name or name == "dt":
            return {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        elif "gradient" in name or "boost" in name or name == "gb":
            return {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5]
            }
        elif "logistic" in name:
            return {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"]
            }
        elif "ridge" in name or "linear" in name:
            return {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        else:
            raise ValueError(f"No default grid for '{model_name}'. Please provide a custom param_grid.")

    # ==============================================================
    # 1. Non-Nested Cross-Validation (Standard GridSearchCV)
    # ==============================================================
    def tune_non_nested(self, estimator, param_grid, X, y, cv=5, scoring=None):
        logger.info("Running GridSearchCV (estimator=%s, cv=%d, param_combinations=%s)",
                    type(estimator).__name__, cv, list(param_grid.keys()))
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid.fit(X, y)

        logger.info("GridSearchCV complete. Best score: %.4f | Optimal params: %s",
                    grid.best_score_, grid.best_params_)

        return {
            "best_estimator": grid.best_estimator_,
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
            "cv_results": pd.DataFrame(grid.cv_results_)
        }

    # ==============================================================
    # 2. Nested Cross-Validation (Unbiased Generalization Estimate)
    # ==============================================================
    def tune_nested_cv(self, estimator, param_grid, X, y, outer_cv=5, inner_cv=3, scoring=None):
        logger.info("Running Nested CV (estimator=%s, outer_cv=%d, inner_cv=%d)",
                    type(estimator).__name__, outer_cv, inner_cv)

        is_classification = (
            y.dtype == "object"
            or str(y.dtype).startswith("cat")
            or (isinstance(y, pd.Series) and y.nunique() <= 20)
            or len(np.unique(y)) <= 20
        )

        if is_classification:
            outer_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
            inner_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
            default_scoring = "accuracy"
        else:
            outer_splitter = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
            inner_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=42)
            default_scoring = "r2"

        active_scoring = scoring or default_scoring

        inner_grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_splitter,
            scoring=active_scoring,
            n_jobs=-1
        )

        nested_scores = cross_val_score(
            inner_grid,
            X,
            y,
            cv=outer_splitter,
            scoring=active_scoring,
            n_jobs=-1
        )

        logger.info("Nested CV outer fold scores: %s", np.round(nested_scores, 4).tolist())
        logger.info("Nested CV generalization score: %.4f (+/- %.4f)", nested_scores.mean(), nested_scores.std())

        final_grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner_splitter, scoring=active_scoring, n_jobs=-1)
        final_grid.fit(X, y)

        return {
            "nested_scores": nested_scores,
            "nested_mean": nested_scores.mean(),
            "nested_std": nested_scores.std(),
            "best_estimator": final_grid.best_estimator_,
            "best_params": final_grid.best_params_,
            "metric": active_scoring
        }

    # ==============================================================
    # 3. Side-by-Side Comparison: Nested vs Non-Nested
    # ==============================================================
    def compare_nested_vs_non_nested(self, estimator, param_grid, X, y, outer_cv=5, inner_cv=3, scoring=None):
        logger.info("Executing comparison between Non-Nested and Nested CV")

        non_nested_res = self.tune_non_nested(estimator, param_grid, X, y, cv=outer_cv, scoring=scoring)
        nested_res = self.tune_nested_cv(estimator, param_grid, X, y, outer_cv=outer_cv, inner_cv=inner_cv, scoring=scoring)

        comparison_df = pd.DataFrame([
            {
                "Method": "Non-Nested CV (GridSearchCV)",
                "Score": f"{non_nested_res['best_score']:.4f}",
                "Type": "Optimized deployment score"
            },
            {
                "Method": "Nested CV (Outer + Inner)",
                "Score": f"{nested_res['nested_mean']:.4f} (+/- {nested_res['nested_std']:.4f})",
                "Type": "Unbiased generalization estimate"
            }
        ])

        logger.info("Comparison summary:\n%s", comparison_df.to_string(index=False))

        return {
            "comparison": comparison_df,
            "non_nested": non_nested_res,
            "nested": nested_res
        }
