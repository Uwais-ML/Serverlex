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


class _LoadDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            # Invoked as class method: ModelDeployer.load("bundle.joblib")
            def _load_as_class(filepath="model_bundle.joblib"):
                bundle = joblib.load(filepath)
                deployer = owner(
                    pipeline=bundle.get("pipeline"),
                    model=bundle.get("model"),
                    metadata=bundle.get("metadata", {})
                )
                logger.info("Loaded model artifact bundle from '%s'", filepath)
                return deployer
            return _load_as_class
        else:
            # Invoked as instance method: loaded_deployer.load("bundle.joblib")
            def _load_as_instance(filepath="model_bundle.joblib"):
                bundle = joblib.load(filepath)
                instance.pipeline = bundle.get("pipeline")
                instance.model = bundle.get("model")
                instance.metadata = bundle.get("metadata", {})
                logger.info("Loaded model artifact bundle from '%s'", filepath)
                return instance
            return _load_as_instance


class ModelDeployer:
    """
    Production Model Serving Engine:
    - Bundles fitted MLPipeline + trained Model into a single deployable artifact.
    - Saves and loads via joblib. Supports both class-level and instance-level loading.
    - Supports predict() directly on raw, uncleaned data (DataFrames or JSON/dicts).
    """

    load = _LoadDescriptor()

    def __init__(self, pipeline=None, model=None, metadata=None):
        if isinstance(pipeline, str) and model is None:
            # Direct instantiation with filepath: ModelDeployer("bundle.joblib")
            bundle = joblib.load(pipeline)
            self.pipeline = bundle.get("pipeline")
            self.model = bundle.get("model")
            self.metadata = bundle.get("metadata", {})
            logger.info("Loaded model artifact bundle from '%s'", pipeline)
            return

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
        if self.pipeline is None or self.model is None:
            raise AttributeError("Pipeline and model must be loaded before calling predict().")

        df = self._prepare_input(raw_input)
        transformed_res = self.pipeline.transform(df)

        if isinstance(transformed_res, tuple):
            X = transformed_res[0]
        else:
            X = transformed_res

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, raw_input):
        if self.pipeline is None or self.model is None:
            raise AttributeError("Pipeline and model must be loaded before calling predict_proba().")

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
