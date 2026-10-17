import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


class ModelExplainer:
    """
    Model Explainability & Feature Importance:
    - Native feature importances (Tree-based & Linear models).
    - Model-agnostic Permutation Importance.
    - Publication-ready bar plots of top driving features.
    """

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None

    def get_feature_importances(self):
        """
        Extracts native feature importances or linear coefficients.
        Returns a sorted pandas DataFrame.
        """
        # 1. Tree-based models (RandomForest, DecisionTree, GradientBoosting)
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            names = self.feature_names or [f"Feature_{i}" for i in range(len(importances))]
            df = pd.DataFrame({"Feature": names, "Importance": importances})
            return df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # 2. Linear models (LogisticRegression, Ridge, LinearRegression)
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)  # Multi-class or 2D coefs
            else:
                coef = np.abs(coef)
            names = self.feature_names or [f"Feature_{i}" for i in range(len(coef))]
            df = pd.DataFrame({"Feature": names, "Importance": coef})
            return df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

        else:
            raise AttributeError("Model does not have native feature_importances_ or coef_. Use permutation_importance() instead.")

    def compute_permutation_importance(self, X, y, scoring=None, n_repeats=10, random_state=42):
        """
        Calculates Model-Agnostic Permutation Importance:
        Measures the drop in model score when a feature's values are randomly shuffled.
        """
        print("Computing Permutation Importance across features...")
        res = permutation_importance(
            self.model,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        names = self.feature_names or (X.columns.tolist() if hasattr(X, "columns") else [f"Feature_{i}" for i in range(X.shape[1])])
        df = pd.DataFrame({
            "Feature": names,
            "Importance_Mean": res.importances_mean,
            "Importance_Std": res.importances_std,
            # Documented column names in DOCUMENTATION.md (1.3.0)
            "Permutation_Importance_Mean": res.importances_mean,
            "Permutation_Importance_Std": res.importances_std,
        }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

        return df

    def plot_feature_importance(self, top_n=10, title="Top Feature Importances", save_path=None):
        """Plots a horizontal bar chart of the top N most important features."""
        try:
            df = self.get_feature_importances().head(top_n)
        except AttributeError:
            print("Cannot extract native feature importance. Run compute_permutation_importance() first.")
            return

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
        ax.barh(df["Feature"], df["Importance"], color="teal", edgecolor="black")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Feature importance plot saved to '{save_path}'")

        plt.show()
