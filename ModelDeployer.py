import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

logger=logging.getLogger("ModelDeployer")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class ModelDeployer:
    """
    Production Model Serving Engine:
    - Bundles fitted MLPipeline + trained Model into a single deployable artifact.
    - Saves and loads via joblib.
    - Supports predict() directly on raw, uncleaned data (DataFrames or JSON/dicts).
    """

    def __init__(self, pipeline=None, model=None, metadata=None):
        self.pipeline = pipeline
        self.model = model
        self.metadata = metadata or {}
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = str(datetime.now())

    def save(self, filepath="model_bundle.joblib"):
        if self.pipeline is None or self.model is None:
            raise ValueError("Both pipeline and model must be provided before saving.")

        bundle = {
            "pipeline": self.pipeline,
            "model": self.model,
            "metadata": self.metadata
        }
        joblib.dump(bundle, filepath)
        logger.info("Saved model artifact bundle to '%s'", filepath)

    @classmethod
    def load(cls, filepath="model_bundle.joblib"):
        bundle = joblib.load(filepath)
        deployer = cls(
            pipeline=bundle["pipeline"],
            model=bundle["model"],
            metadata=bundle.get("metadata", {})
        )
        logger.info("Loaded model artifact bundle from '%s'", filepath)
        return deployer

    def _prepare_input(self, raw_input):
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        else:
            raise TypeError("Input must be a pandas DataFrame, a dict (single sample), or a list of dicts.")
        return df

    def predict(self, raw_input):
        df = self._prepare_input(raw_input)
        transformed_res = self.pipeline.transform(df)

        if isinstance(transformed_res, tuple):
            X = transformed_res[0]
        else:
            X = transformed_res

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, raw_input):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"Model '{type(self.model).__name__}' does not support predict_proba().")

        df = self._prepare_input(raw_input)
        transformed_res = self.pipeline.transform(df)

        if isinstance(transformed_res, tuple):
            X = transformed_res[0]
        else:
            X = transformed_res

        probabilities = self.model.predict_proba(X)
        return probabilities
