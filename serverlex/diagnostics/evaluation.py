import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

logger=logging.getLogger("Evaluator")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class ModelEvaluator:
    def __init__(self):
        self.metrics={}
        self.y_true=None
        self.y_pred=None
        self.y_prob=None
        self.task=None

    # ==============================================================
    # 1. Classification Diagnostics
    # ==============================================================
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        try:
            cm=confusion_matrix(y_true, y_pred, labels=labels)
            fig, ax=plt.subplots(figsize=(6, 5))
            cax=ax.imshow(cm, cmap="Blues")
            fig.colorbar(cax, ax=ax)

            tick_marks=range(len(cm))
            display_labels=labels if labels is not None else [f"Class {i}" for i in tick_marks]
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(display_labels, rotation=45, ha="right")
            ax.set_yticklabels(display_labels)

            thresh=cm.max() / 2.0
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black", fontsize=11, fontweight="bold")

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel("Actual Label")
            ax.set_xlabel("Predicted Label")
            plt.tight_layout()
            plt.show()
            return cm
        except Exception as e:
            logger.error("Error in plot_confusion_matrix: %s", e)

    def plot_roc_curve(self, y_true, y_prob, title="ROC Curve"):
        try:
            y_true_arr = np.asarray(y_true)
            y_prob_arr = np.asarray(y_prob)

            fig, ax = plt.subplots(figsize=(6, 5))

            if y_prob_arr.ndim > 1 and y_prob_arr.shape[1] > 2:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_true_arr)
                y_bin = label_binarize(y_true_arr, classes=classes)
                auc_scores = []
                for idx, cls_name in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob_arr[:, idx])
                    score = float(auc(fpr, tpr))
                    auc_scores.append(score)
                    ax.plot(fpr, tpr, lw=1.5, label=f"Class {cls_name} (AUC={score:.3f})")
                roc_auc = float(np.mean(auc_scores))
                ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
                ax.set_title(f"{title} (Macro Avg AUC = {roc_auc:.4f})", fontsize=12, fontweight="bold")
            else:
                prob_vec = y_prob_arr[:, 1] if (y_prob_arr.ndim > 1 and y_prob_arr.shape[1] == 2) else y_prob_arr
                fpr, tpr, _ = roc_curve(y_true_arr, prob_vec)
                roc_auc = float(auc(fpr, tpr))
                ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
                ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
                ax.set_title(title, fontsize=12, fontweight="bold")

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
            logger.info("ROC-AUC score: %.4f", roc_auc)
            return roc_auc
        except Exception as e:
            logger.error("Error in plot_roc_curve: %s", e)
            raise

    def plot_precision_recall(self, y_true, y_prob, title="Precision-Recall Curve"):
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            fig, ax=plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, color="purple", lw=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error("Error in plot_precision_recall: %s", e)

    # ==============================================================
    # 2. Regression Diagnostics
    # ==============================================================
    def plot_actual_vs_predicted(self, y_true, y_pred, title="Actual vs. Predicted"):
        try:
            fig, ax=plt.subplots(figsize=(6, 5))
            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", color="royalblue")

            min_val=min(min(y_true), min(y_pred))
            max_val=max(max(y_true), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], color="crimson", lw=2, linestyle="--", label="Ideal Fit (y=x)")

            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error("Error in plot_actual_vs_predicted: %s", e)

    def plot_residuals(self, y_true, y_pred, title="Residual Analysis"):
        try:
            residuals = np.array(y_true) - np.array(y_pred)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

            ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", color="darkcyan")
            ax1.axhline(0, color="crimson", linestyle="--", lw=1.5)
            ax1.set_xlabel("Predicted Values")
            ax1.set_ylabel("Residuals")
            ax1.set_title("Residuals vs. Predicted", fontsize=11, fontweight="bold")
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.hist(residuals, bins=20, color="teal", edgecolor="black", alpha=0.7)
            ax2.axvline(0, color="crimson", linestyle="--", lw=1.5)
            ax2.set_xlabel("Residual Value")
            ax2.set_ylabel("Count")
            ax2.set_title("Distribution of Errors", fontsize=11, fontweight="bold")

            plt.suptitle(title, fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error("Error in plot_residuals: %s", e)

    # ==============================================================
    # 3. Deep End-to-End Diagnostics Runner
    # ==============================================================
    def evaluate(self, model, X, y):
        try:
            self.y_true = np.array(y)
            self.y_pred = model.predict(X)

            is_class = (
                self.y_true.dtype == "object"
                or str(self.y_true.dtype).startswith("cat")
                or len(np.unique(self.y_true)) <= 20
            )
            self.task = "Classification" if is_class else "Regression"

            logger.info("Evaluating model predictions (task=%s, samples=%d)", self.task, len(self.y_true))

            if is_class:
                logger.info("Classification report:\n%s", classification_report(self.y_true, self.y_pred))
                self.plot_confusion_matrix(self.y_true, self.y_pred)

                if len(np.unique(self.y_true)) == 2 and hasattr(model, "predict_proba"):
                    self.y_prob = model.predict_proba(X)[:, 1]
                    self.plot_roc_curve(self.y_true, self.y_prob)
                    self.plot_precision_recall(self.y_true, self.y_prob)

            else:
                r2 = r2_score(self.y_true, self.y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
                mae = mean_absolute_error(self.y_true, self.y_pred)
                mape = mean_absolute_percentage_error(self.y_true, self.y_pred)

                n = len(self.y_true)
                p = X.shape[1]
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else r2

                logger.info("Regression metrics: R2=%.4f, Adj_R2=%.4f, RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%",
                            r2, adj_r2, rmse, mae, mape * 100)

                self.plot_actual_vs_predicted(self.y_true, self.y_pred)
                self.plot_residuals(self.y_true, self.y_pred)

        except Exception as e:
            logger.error("Error in evaluate: %s", e)
