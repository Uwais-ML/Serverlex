import logging
import hashlib
import json
import os
import re
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


import hmac

class _LoadDescriptor:
    def __get__(self, instance, owner):
        if instance is None:
            # Invoked as class method: ModelDeployer.load("bundle.joblib")
            def _load_as_class(filepath="model_bundle.joblib", verify_hash=None, hmac_key=None):
                has_sidecar = os.path.exists(filepath + ".sha256") or os.path.exists(filepath + ".hmac")
                should_verify = verify_hash is True or (verify_hash is None and has_sidecar) or (hmac_key is not None)
                if should_verify:
                    owner._verify_bundle_integrity(filepath, hmac_key=hmac_key)
                try:
                    bundle = joblib.load(filepath)
                    if not isinstance(bundle, dict):
                        raise ValueError(f"Expected dict bundle artifact, got {type(bundle)}")
                except Exception as e:
                    raise ValueError(f"Corrupt or invalid model bundle '{filepath}': {e}") from e

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
            def _load_as_instance(filepath="model_bundle.joblib", verify_hash=None, hmac_key=None):
                has_sidecar = os.path.exists(filepath + ".sha256") or os.path.exists(filepath + ".hmac")
                should_verify = verify_hash is True or (verify_hash is None and has_sidecar) or (hmac_key is not None)
                if should_verify:
                    instance._verify_bundle_integrity(filepath, hmac_key=hmac_key)
                try:
                    bundle = joblib.load(filepath)
                    if not isinstance(bundle, dict):
                        raise ValueError(f"Expected dict bundle artifact, got {type(bundle)}")
                except Exception as e:
                    raise ValueError(f"Corrupt or invalid model bundle '{filepath}': {e}") from e

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
    - Strict Mode: When strict=True, raises ValueError if required features are missing, unseen categories are passed, or non-numeric values are given to numeric fields.
    - Cryptographic Bundle Verification: SHA-256 integrity signature and HMAC authenticity check.
    """

    load = _LoadDescriptor()

    def __init__(self, pipeline=None, model=None, metadata=None):
        if isinstance(pipeline, str) and model is None:
            # Direct instantiation with filepath: ModelDeployer("bundle.joblib")
            filepath = pipeline
            if os.path.exists(filepath + ".sha256") or os.path.exists(filepath + ".hmac"):
                self._verify_bundle_integrity(filepath)
            try:
                bundle = joblib.load(filepath)
                if not isinstance(bundle, dict):
                    raise ValueError(f"Expected dict bundle artifact, got {type(bundle)}")
            except Exception as e:
                raise ValueError(f"Corrupt or invalid model bundle '{filepath}': {e}") from e

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

    @staticmethod
    def _compute_file_hmac(filepath, key):
        if isinstance(key, str):
            key = key.encode('utf-8')
        mac = hmac.new(key, digestmod=hashlib.sha256)
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                mac.update(chunk)
        return mac.hexdigest()

    @classmethod
    def _verify_bundle_integrity(cls, filepath, hmac_key=None):
        if hmac_key is not None:
            meta_path = filepath + ".hmac"
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Authenticity check failed: missing HMAC signature file '{meta_path}'")
            with open(meta_path, 'r') as f:
                expected_hmac = f.read().strip()
            actual_hmac = cls._compute_file_hmac(filepath, hmac_key)
            if not hmac.compare_digest(actual_hmac, expected_hmac):
                raise ValueError(f"Bundle authenticity verification failed! HMAC mismatch for '{filepath}'")
            logger.info("Bundle authenticity verified successfully via HMAC-SHA256")
            return

        if os.path.exists(filepath + ".hmac") and hmac_key is None:
            raise ValueError(f"Bundle '{filepath}' was saved with HMAC authenticity protection (.hmac sidecar found). Please provide 'hmac_key' to load and verify it.")

        meta_path = filepath + ".sha256"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Integrity check failed: missing SHA-256 signature file '{meta_path}'")
        with open(meta_path, 'r') as f:
            expected_hash = f.read().strip()
        actual_hash = cls._compute_file_hash(filepath)
        if actual_hash != expected_hash:
            raise ValueError(f"Bundle integrity verification failed! Signature mismatch for '{filepath}'")
        logger.info("Bundle integrity verified successfully via SHA-256")

    def save(self, filepath="model_bundle.joblib", compute_hash=False, hmac_key=None):
        if self.pipeline is None or self.model is None:
            raise ValueError("Both pipeline and model must be provided before saving.")

        bundle = {
            "pipeline": self.pipeline,
            "model": self.model,
            "metadata": self.metadata
        }
        joblib.dump(bundle, filepath)
        logger.info("Saved model artifact bundle to '%s'", filepath)

        if hmac_key is not None:
            file_hmac = self._compute_file_hmac(filepath, hmac_key)
            with open(filepath + ".hmac", 'w') as f:
                f.write(file_hmac)
            logger.info("Generated HMAC-SHA256 authenticity signature")
        elif compute_hash:
            file_hash = self._compute_file_hash(filepath)
            with open(filepath + ".sha256", 'w') as f:
                f.write(file_hash)
            logger.info("Generated SHA-256 integrity signature: %s", file_hash)

    def _validate_and_sanitize(self, df, strict=False):
        """
        Validates incoming inference schema, performs safe type coercion without wiping currency strings,
        and auto-imputes missing keys. In strict mode, verifies unseen categories and invalid types.
        """
        clean_df = df.copy()

        # 1. Standardize incoming column names to match Clean.clean_column_names
        clean_df.columns = [
            re.sub(r"[^\w_]", "", c.strip().lower().replace(' ', '_'))
            for c in clean_df.columns
        ]

        # 1b. Expand date_col into sub-features (mirrors pipeline.transform) so users
        #     can pass e.g. signup_date instead of signup_date_year/month/day/dayofweek.
        date_col = getattr(self.pipeline, "date_col", None) if self.pipeline else None
        if date_col and date_col in clean_df.columns:
            dt = pd.to_datetime(clean_df[date_col], format='mixed', errors='coerce')
            clean_df[f'{date_col}_year'] = dt.dt.year
            clean_df[f'{date_col}_month'] = dt.dt.month
            clean_df[f'{date_col}_day'] = dt.dt.day
            clean_df[f'{date_col}_dayofweek'] = dt.dt.dayofweek
            clean_df = clean_df.drop(columns=[date_col])

        # 2. Strict mode validation checks before coercion/imputation
        if strict and self.pipeline is not None:
            problems = []

            # Check unseen categorical levels
            if hasattr(self.pipeline, "cat_categories") and self.pipeline.cat_categories:
                for col, cats in self.pipeline.cat_categories.items():
                    if col in clean_df.columns:
                        norm = clean_df[col].dropna().astype(str).str.lower().str.replace(' ', '')
                        unseen = sorted(set(norm) - set(cats))
                        if unseen:
                            problems.append(f"{col}: unseen categories {unseen}")

            # Check non-numeric values in numeric columns
            if hasattr(self.pipeline, "num_fill") and self.pipeline.num_fill:
                for col in self.pipeline.num_fill.keys():
                    if col in clean_df.columns:
                        raw = clean_df[col]
                        stripped = raw.astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
                        coerced = pd.to_numeric(stripped, errors='coerce')
                        if (coerced.isna() & raw.notna()).any():
                            problems.append(f"{col}: non-numeric values")

            # Check missing required training features
            date_col = getattr(self.pipeline, "date_col", None)
            generated_date_cols = {f"{date_col}_year", f"{date_col}_month", f"{date_col}_day", f"{date_col}_dayofweek"} if date_col else set()

            missing_features = []
            if hasattr(self.pipeline, "num_fill"):
                missing_num = [c for c in self.pipeline.num_fill.keys() if c not in clean_df.columns]
                if date_col and date_col in clean_df.columns:
                    missing_num = [c for c in missing_num if c not in generated_date_cols]
                missing_features.extend(missing_num)

            if hasattr(self.pipeline, "cat_fill"):
                missing_features.extend([c for c in self.pipeline.cat_fill.keys() if c not in clean_df.columns])

            if missing_features:
                problems.append(f"missing required training features: {missing_features}")

            if problems:
                raise ValueError("Strict deployment error: " + "; ".join(problems))

        # 3. Handle numerical missing columns and currency parsing
        # Note: date_col sub-features were already expanded above; the source date col is gone.
        generated_date_cols = {f"{date_col}_year", f"{date_col}_month", f"{date_col}_day", f"{date_col}_dayofweek"} if date_col else set()

        if hasattr(self.pipeline, "num_fill") and self.pipeline.num_fill:
            for num_col, fill_val in self.pipeline.num_fill.items():
                if num_col not in clean_df.columns:
                    if num_col in generated_date_cols:
                        # Already expanded from date_col above; skip imputation
                        continue
                    clean_df[num_col] = fill_val
                    logger.info("Missing numerical feature '%s' imputed with training baseline: %s", num_col, fill_val)
                else:
                    col_vals = clean_df[num_col]
                    if col_vals.dtype == object or str(col_vals.dtype).startswith('str'):
                        stripped = col_vals.astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
                        clean_df[num_col] = pd.to_numeric(stripped, errors='coerce').fillna(fill_val)


        # 4. Handle categorical missing columns
        if hasattr(self.pipeline, "cat_fill") and self.pipeline.cat_fill:
            for cat_col, fill_val in self.pipeline.cat_fill.items():
                if cat_col not in clean_df.columns:
                    clean_df[cat_col] = fill_val
                    logger.info("Missing categorical feature '%s' imputed with training baseline: %s", cat_col, fill_val)

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

    def predict(self, raw_input, strict=False, threshold=None):
        if self.pipeline is None or self.model is None:
            raise AttributeError("Pipeline and model must be loaded before calling predict().")

        df = self._prepare_input(raw_input, strict=strict)
        transformed_res = self.pipeline.transform(df)

        if isinstance(transformed_res, tuple):
            X = transformed_res[0]
        else:
            X = transformed_res

        # If a custom decision threshold is provided and model supports probabilities
        if threshold is not None and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return (proba[:, 1] >= threshold).astype(int)

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
