import logging
import hashlib
import json
import os
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
            def _load_as_class(filepath="model_bundle.joblib", verify_hash=False):
                if verify_hash:
                    owner._verify_bundle_integrity(filepath)
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
            def _load_as_instance(filepath="model_bundle.joblib", verify_hash=False):
                if verify_hash:
                    instance._verify_bundle_integrity(filepath)
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
    - Saves and loads via joblib supporting class-level, instance-level, and constructor loading.
    - Schema Validation: Auto-imputes missing inference keys using learned training pipeline statistics.
    - Currency & Type Resilience: Preserves currency strings for pipeline cleaning rather than wiping them.
    - Strict Mode: When strict=True, raises ValueError if required features are missing from payload.
    - Cryptographic Bundle Verification: Optional SHA-256 integrity check.
    """

    load = _LoadDescriptor()

    def __init__(self, pipeline=None, model=None, metadata=None):
        if isinstance(pipeline, str) and model is None:
            # Direct instantiation with filepath: ModelDeployer("bundle.joblib")
            filepath = pipeline
            bundle = joblib.load(filepath)
            self.pipeline = bundle.get("pipeline")
            self.model = bundle.get("model")
            self.metadata = bundle.get("metadata", {})
            logger.info("Loaded model artifact bundle from '%s'", filepath)
            return

        self.pipeline = pipeline
        self.model = model
        self.metadata = metadata or {}
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = str(datetime.now())

    @staticmethod
    def _compute_file_hash(filepath):
        sha = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    @classmethod
    def _verify_bundle_integrity(cls, filepath):
        meta_path = filepath + ".sha256"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Integrity check failed: missing SHA-256 signature file '{meta_path}'")
        with open(meta_path, 'r') as f:
            expected_hash = f.read().strip()
        actual_hash = cls._compute_file_hash(filepath)
        if actual_hash != expected_hash:
            raise ValueError(f"Bundle integrity verification failed! Signature mismatch for '{filepath}'")
        logger.info("Bundle integrity verified successfully via SHA-256")

    def save(self, filepath="model_bundle.joblib", compute_hash=False):
        if self.pipeline is None or self.model is None:
            raise ValueError("Both pipeline and model must be provided before saving.")

        bundle = {
            "pipeline": self.pipeline,
            "model": self.model,
            "metadata": self.metadata
        }
        joblib.dump(bundle, filepath)
        logger.info("Saved model artifact bundle to '%s'", filepath)

        if compute_hash:
            file_hash = self._compute_file_hash(filepath)
            with open(filepath + ".sha256", 'w') as f:
                f.write(file_hash)
            logger.info("Generated SHA-256 integrity signature: %s", file_hash)

    def _validate_and_sanitize(self, df, strict=False):
        """
        Validates incoming inference schema, performs safe type coercion without wiping currency strings,
        and auto-imputes missing keys.
        """
        clean_df = df.copy()

        # 1. Snake case incoming column names
        clean_df.columns = [c.strip().lower().replace(' ', '_').replace('$', '') for c in clean_df.columns]

        # 2. Check for missing columns learned during pipeline training
        missing_num = []
        if hasattr(self.pipeline, "num_fill") and self.pipeline.num_fill:
            for num_col, fill_val in self.pipeline.num_fill.items():
                if num_col not in clean_df.columns:
                    missing_num.append(num_col)
                    clean_df[num_col] = fill_val
                    logger.info("Missing numerical feature '%s' imputed with training baseline: %s", num_col, fill_val)
                else:
                    # Clean currency/comma strings BEFORE numeric coercion so values like '$98.50' aren't wiped
                    col_vals = clean_df[num_col]
                    if col_vals.dtype == object or str(col_vals.dtype).startswith('str'):
                        # If string contains currency or numbers, strip currency symbol
                        stripped = col_vals.astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
                        # Only convert if actually numeric
                        clean_df[num_col] = pd.to_numeric(stripped, errors='coerce').fillna(fill_val)

        missing_cat = []
        if hasattr(self.pipeline, "cat_fill") and self.pipeline.cat_fill:
            for cat_col, fill_val in self.pipeline.cat_fill.items():
                if cat_col not in clean_df.columns:
                    missing_cat.append(cat_col)
                    clean_df[cat_col] = fill_val
                    logger.info("Missing categorical feature '%s' imputed with training baseline: %s", cat_col, fill_val)

        all_missing = missing_num + missing_cat
        if strict and all_missing:
            raise ValueError(f"Strict deployment error: inference input missing required training features: {all_missing}")

        return clean_df

    def _prepare_input(self, raw_input, strict=False):
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        elif isinstance(raw_input, list):
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, pd.DataFrame):
            df = raw_input.copy()
        else:
            raise TypeError("Input must be a pandas DataFrame, a dict (single sample), or a list of dicts.")

        return self._validate_and_sanitize(df, strict=strict)

    def predict(self, raw_input, strict=False):
        if self.pipeline is None or self.model is None:
            raise AttributeError("Pipeline and model must be loaded before calling predict().")

        df = self._prepare_input(raw_input, strict=strict)
        transformed_res = self.pipeline.transform(df)

        if isinstance(transformed_res, tuple):
            X = transformed_res[0]
        else:
            X = transformed_res

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, raw_input, strict=False):
        if self.pipeline is None or self.model is None:
            raise AttributeError("Pipeline and model must be loaded before calling predict_proba().")

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"Model '{type(self.model).__name__}' does not support predict_proba().")

        df = self._prepare_input(raw_input, strict=strict)
        transformed_res = self.pipeline.transform(df)

        if isinstance(transformed_res, tuple):
            X = transformed_res[0]
        else:
            X = transformed_res

        probabilities = self.model.predict_proba(X)
        return probabilities
