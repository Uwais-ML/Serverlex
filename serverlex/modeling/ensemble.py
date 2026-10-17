import logging
import pandas as pd
import numpy as np

from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

logger=logging.getLogger("EnsembleBuilder")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class EnsembleBuilder:
    """
    Automated Model Ensembling (Voting & Stacking):
    Combines diverse algorithms to improve predictive performance and reduce variance.
    """

    def __init__(self, task="classification"):
        self.task=task.lower()

    def build_voting_ensemble(self, estimators, voting="soft"):
        """
        Builds a Voting Ensemble:
        Averages predictions across multiple models (soft = probability-weighted, hard = majority vote).
        estimators: list of (name, model) tuples, e.g. [('rf', rf), ('gb', gb), ('lr', lr)]
        """
        try:
            logger.info("Building Voting Ensemble (%d estimators, mode=%s)", len(estimators), voting)
            if self.task == "classification":
                ensemble = VotingClassifier(estimators=estimators, voting=voting)
            else:
                ensemble = VotingRegressor(estimators=estimators)
            return ensemble
        except Exception as e:
            logger.error("Error in build_voting_ensemble: %s", e)

    def build_stacking_ensemble(self, estimators, final_estimator=None, cv=5):
        """
        Builds a Stacking Ensemble:
        Uses a meta-learner to learn how to best combine predictions from base models.
        """
        try:
            logger.info("Building Stacking Ensemble (%d base estimators, cv=%d)", len(estimators), cv)
            if self.task == "classification":
                meta = final_estimator or LogisticRegression(max_iter=1000)
                ensemble = StackingClassifier(estimators=estimators, final_estimator=meta, cv=cv, n_jobs=-1)
            else:
                meta = final_estimator or Ridge()
                ensemble = StackingRegressor(estimators=estimators, final_estimator=meta, cv=cv, n_jobs=-1)
            return ensemble
        except Exception as e:
            logger.error("Error in build_stacking_ensemble: %s", e)

    def evaluate_ensemble(self, ensemble, X, y, cv=5):
        """Evaluates an ensemble model using cross-validation."""
        try:
            if not isinstance(y, pd.Series):
                y = pd.Series(y)

            if self.task == "classification":
                kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                acc = cross_val_score(ensemble, X, y, cv=kf, scoring="accuracy")
                f1 = cross_val_score(ensemble, X, y, cv=kf, scoring="f1_weighted")
                logger.info("Ensemble CV Accuracy: %.4f (+/- %.4f) | F1: %.4f", acc.mean(), acc.std(), f1.mean())
                return {"Accuracy": round(acc.mean(), 4), "Std_Accuracy": round(acc.std(), 4), "F1": round(f1.mean(), 4)}
            else:
                kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                r2 = cross_val_score(ensemble, X, y, cv=kf, scoring="r2")
                rmse = cross_val_score(ensemble, X, y, cv=kf, scoring="neg_root_mean_squared_error")
                logger.info("Ensemble CV R2: %.4f (+/- %.4f) | RMSE: %.4f", r2.mean(), r2.std(), -rmse.mean())
                return {"R2": round(r2.mean(), 4), "Std_R2": round(r2.std(), 4), "RMSE": round(-rmse.mean(), 4)}
        except Exception as e:
            logger.error("Error in evaluate_ensemble: %s", e)
