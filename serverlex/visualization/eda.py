import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Clean import Clean

logger=logging.getLogger("Visualization")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class Visualize:
    def __init__(self, data=None):
        """
        data can be:
        - A CSV file path (str)
        - An instance of Clean
        - A pandas DataFrame
        - None (data can be set or loaded later)
        """
        self.data = None
        self.cleaner = None
        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        """Sets or updates the active dataset."""
        if isinstance(data, str):
            self.cleaner = Clean(data)
            self.cleaner.load()
            self.data = self.cleaner.data
        elif isinstance(data, Clean):
            self.cleaner = data
            if not hasattr(self.cleaner, "data") or self.cleaner.data is None:
                self.cleaner.load()
            self.data = self.cleaner.data
        elif isinstance(data, pd.DataFrame):
            self.cleaner = Clean.__new__(Clean)
            self.cleaner.data = data.copy()
            self.cleaner.path = None
            self.data = data

    def _sync(self):
        """Syncs self.data from self.cleaner if cleaner modified the dataframe."""
        if self.cleaner is not None and hasattr(self.cleaner, "data") and self.cleaner.data is not None:
            self.data = self.cleaner.data

    # ==========================
    # 1. Individual Plot Methods
    # ==========================

    def plot_missing(self, title="Missing Values per Column"):
        """Plots a bar chart showing the count of missing values for columns with NaNs."""
        self._sync()
        try:
            if self.data is None:
                print("No data available to plot.")
                return

            missing = self.data.isnull().sum()
            missing = missing[missing > 0]
            if missing.empty:
                print("No missing values found in dataset.")
                return

            fig, ax = plt.subplots(figsize=(8, 4))
            missing.plot(kind="bar", color="salmon", edgecolor="black", ax=ax)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel("Count of NaNs")
            ax.set_xlabel("Columns")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(e)

    def plot_distribution(self, column=None, bins=20):
        """Plots histograms for one or all numerical columns."""
        self._sync()
        try:
            if self.data is None:
                print("No data available to plot.")
                return

            num_cols = self.data.select_dtypes(include="number").columns
            if len(num_cols) == 0:
                print("No numerical columns available.")
                return

            cols = [column] if column else num_cols
            n_cols = len(cols)
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), squeeze=False)
            for i, col in enumerate(cols):
                if col in self.data.columns:
                    axes[0, i].hist(self.data[col].dropna(), bins=bins, color="skyblue", edgecolor="black")
                    axes[0, i].set_title(f"Dist: {col}")
                    axes[0, i].set_xlabel(col)
                    axes[0, i].set_ylabel("Frequency")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(e)

    def plot_box(self, columns=None, title="Outlier Detection (Box Plot)"):
        """Plots box plots for numeric columns to visualize outliers."""
        self._sync()
        try:
            if self.data is None:
                print("No data available to plot.")
                return

            num_df = self.data.select_dtypes(include="number")
            if num_df.empty:
                print("No numerical columns for box plot.")
                return

            if columns:
                valid_cols = [c for c in columns if c in num_df.columns]
                num_df = num_df[valid_cols]

            fig, ax = plt.subplots(figsize=(max(6, len(num_df.columns) * 1.5), 5))
            num_df.boxplot(ax=ax, patch_artist=True, boxprops=dict(facecolor="lightblue"))
            ax.set_title(title, fontsize=12, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(e)

    def plot_categorical(self, column=None, top_n=10, title=None):
        """Plots horizontal bar chart of top N categories."""
        self._sync()
        try:
            if self.data is None:
                print("No data available to plot.")
                return

            cat_cols = self.data.select_dtypes(include=["object", "category"]).columns
            if len(cat_cols) == 0:
                print("No categorical columns available.")
                return

            col = column if column else cat_cols[0]
            if col not in self.data.columns:
                print(f"Column '{col}' not found.")
                return

            counts = self.data[col].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(8, 4))
            counts.plot(kind="barh", color="mediumaquamarine", edgecolor="black", ax=ax)
            ax.invert_yaxis()
            ax.set_title(title or f"Top {top_n} Categories in '{col}'", fontsize=12, fontweight="bold")
            ax.set_xlabel("Count")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(e)

    def plot_correlation(self, title="Correlation Matrix Heatmap"):
        """Plots a correlation heatmap for numerical features using matplotlib."""
        self._sync()
        try:
            if self.data is None:
                print("No data available to plot.")
                return

            num_df = self.data.select_dtypes(include="number")
            if num_df.shape[1] < 2:
                print("Need at least 2 numerical columns for correlation plot.")
                return

            corr = num_df.corr()
            fig, ax = plt.subplots(figsize=(max(6, len(corr) * 1.2), max(5, len(corr) * 1.0)))
            cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax, ax=ax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)

            # Annotate correlation values in cells
            for i in range(len(corr)):
                for j in range(len(corr.columns)):
                    val = corr.iloc[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.7 else "white", fontsize=9)

            ax.set_title(title, fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(e)

    def plot_scatter(self, x, y, title=None):
        """Plots a scatter plot between two numerical columns."""
        self._sync()
        try:
            if self.data is None:
                print("No data available to plot.")
                return

            if x not in self.data.columns or y not in self.data.columns:
                print(f"One or both columns ({x}, {y}) not in DataFrame.")
                return

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(self.data[x], self.data[y], color="steelblue", alpha=0.7, edgecolors="k")
            ax.set_title(title or f"{x} vs {y}", fontsize=12, fontweight="bold")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(e)

    # ==============================================================
    # 2. Live Interactive Cleaning Pipeline with plt.ion()
    # ==============================================================

    def live_clean(
        self,
        cleaner=None,
        data_time=None,
        numeric_string_cols=None,
        missing_param="m",
        outlier_method="IQR",
        outlier_param="m",
        corr_target=None,
        corr_threshold=0.1,
        pause_time=1.5
    ):
        """
        Runs the cleaning pipeline step-by-step with plt.ion() (interactive mode),
        updating live plots at each stage to show before & after transformations.
        """
        if cleaner is not None:
            self.cleaner = cleaner
        elif self.cleaner is None:
            print("No cleaner instance provided.")
            return

        # Enable interactive plotting
        plt.ion()
        fig = plt.figure(figsize=(10, 6))
        logger.info("Initializing live visual cleaning session")

        try:
            # 1. Load & Deduplicate
            self.cleaner.load()
            self.cleaner.clean_column_names()
            self.cleaner.delete_duplicates()
            self._sync()

            # Step 1: Missing values BEFORE imputation
            plt.clf()
            ax = fig.add_subplot(111)
            missing_before = self.data.isnull().sum()
            missing_before = missing_before[missing_before > 0]
            if not missing_before.empty:
                missing_before.plot(kind="bar", color="crimson", edgecolor="black", ax=ax)
                ax.set_title("Step 1: Missing Values (BEFORE Imputation)", fontsize=13, fontweight="bold")
                ax.set_ylabel("Count of NaNs")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.draw()
                plt.pause(pause_time)

            # Step 2: Impute Missing Values
            if numeric_string_cols:
                self.cleaner.clean_numeric_strings(numeric_string_cols)
            self.cleaner.Clean_string(data_time=data_time)
            self.cleaner.fill_missing(param=missing_param)
            self.cleaner.fill_categorical_missing()
            self._sync()

            plt.clf()
            ax = fig.add_subplot(111)
            missing_after = self.data.isnull().sum()
            missing_after.plot(kind="bar", color="mediumseagreen", edgecolor="black", ax=ax)
            ax.set_title("Step 2: Missing Values (AFTER Imputation - All Filled)", fontsize=13, fontweight="bold")
            ax.set_ylabel("Count of NaNs")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.draw()
            plt.pause(pause_time)

            # Step 3: Outliers BEFORE Handling
            num_cols = self.data.select_dtypes(include="number").columns.tolist()
            if num_cols:
                plt.clf()
                ax = fig.add_subplot(111)
                self.data[num_cols].boxplot(ax=ax, patch_artist=True, boxprops=dict(facecolor="orange"))
                ax.set_title(f"Step 3: Outliers (BEFORE {outlier_method})", fontsize=13, fontweight="bold")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.draw()
                plt.pause(pause_time)

                # Execute outlier handling
                if outlier_method == "IQR":
                    self.cleaner.outliers_IQR(param=outlier_param)
                else:
                    self.cleaner.outliers_Z(param=outlier_param)
                self._sync()

                # Step 4: Outliers AFTER Handling
                plt.clf()
                ax = fig.add_subplot(111)
                self.data[num_cols].boxplot(ax=ax, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
                ax.set_title(f"Step 4: Outliers (AFTER {outlier_method} param='{outlier_param}')", fontsize=13, fontweight="bold")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.draw()
                plt.pause(pause_time)

            # Step 5: Correlation Heatmap BEFORE low_corr
            num_df = self.data.select_dtypes(include="number")
            if num_df.shape[1] >= 2:
                plt.clf()
                ax = fig.add_subplot(111)
                corr = num_df.corr()
                cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                fig.colorbar(cax, ax=ax)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr.columns)
                ax.set_title("Step 5: Correlation Matrix (BEFORE low_corr filter)", fontsize=13, fontweight="bold")
                plt.tight_layout()
                plt.draw()
                plt.pause(pause_time)

                # Execute low_corr
                if corr_threshold is not None:
                    self.cleaner.low_corr(target=corr_target, threshold=corr_threshold)
                    self._sync()

                    # Step 6: Correlation Heatmap AFTER low_corr
                    plt.clf()
                    ax = fig.add_subplot(111)
                    corr_after = self.data.select_dtypes(include="number").corr()
                    cax2 = ax.imshow(corr_after, cmap="coolwarm", vmin=-1, vmax=1)
                    fig.colorbar(cax2, ax=ax)
                    ax.set_xticks(range(len(corr_after.columns)))
                    ax.set_yticks(range(len(corr_after.columns)))
                    ax.set_xticklabels(corr_after.columns, rotation=45, ha="right")
                    ax.set_yticklabels(corr_after.columns)
                    ax.set_title("Step 6: Correlation Matrix (AFTER low_corr filter)", fontsize=13, fontweight="bold")
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(pause_time)

            logger.info("Live visual cleaning session completed")
        except Exception as e:
            logger.error("Error in live_clean: %s", e)
        finally:
            plt.ioff()
            plt.show()
