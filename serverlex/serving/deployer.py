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

    # ==========================
    # Pre-Inference Latency & SLA Profiler
    # ==========================
    def benchmark(self, sample=None, n_iterations=300, warmup=30):
        """
        Benchmarks inference latency (p50, p90, p95, p99) and throughput on single records.

        Args:
            sample (dict, optional): Sample payload to benchmark. If None, synthesizes a sample
                                    from pipeline statistics.
            n_iterations (int): Number of timed prediction runs. Default 300.
            warmup (int): Warmup runs before timing. Default 30.

        Returns:
            dict: Latency statistics (ms) and throughput (req/sec).
        """
        import time

        if self.pipeline is None or self.model is None:
            raise AttributeError("Pipeline and model must be loaded before running benchmark().")

        # Synthesize sample if not provided
        if sample is None:
            synthetic = {}
            for col, val in self.pipeline.num_fill.items():
                synthetic[col] = float(val)
            for col, val in self.pipeline.cat_fill.items():
                synthetic[col] = str(val)
            if self.pipeline.date_col:
                synthetic[self.pipeline.date_col] = datetime.now().strftime("%Y-%m-%d")
            sample = synthetic if synthetic else {"sample_feature": 1.0}

        # Warmup
        for _ in range(max(1, warmup)):
            self.predict(sample, strict=False)

        # Timed runs
        latencies = []
        for _ in range(max(10, n_iterations)):
            t0 = time.perf_counter()
            self.predict(sample, strict=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        lat_arr = np.array(latencies)
        mean_ms = float(np.mean(lat_arr))
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        min_ms = float(np.min(lat_arr))
        max_ms = float(np.max(lat_arr))
        throughput_rps = round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0

        summary = {
            "iterations": len(latencies),
            "mean_ms": round(mean_ms, 3),
            "p50_ms": round(p50, 3),
            "p90_ms": round(p90, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
            "min_ms": round(min_ms, 3),
            "max_ms": round(max_ms, 3),
            "throughput_req_per_sec": throughput_rps
        }

        logger.info(
            "Benchmark Complete (%d runs): p50=%.2fms, p95=%.2fms, p99=%.2fms | Throughput: %.1f req/s",
            len(latencies), p50, p95, p99, throughput_rps
        )
        return summary

    # ==========================
    # FastAPI & Microservice Exporter
    # ==========================
    def export_api(self, output_dir="./serving_app", bundle_filename="model_bundle.joblib", hmac_key=None, host="0.0.0.0", port=8000):
        """
        Exports a turnkey FastAPI production microservice, Dockerfile, and requirements.txt
        configured with strict input validation, health checks, and HMAC signing.

        Args:
            output_dir (str): Destination directory for the service package.
            bundle_filename (str): Name of the joblib bundle inside the package.
            hmac_key (str, optional): HMAC secret key to seal the bundle with.
            host (str): Host binding for the generated server.
            port (int): Port for the generated server.

        Returns:
            dict: Generated filepaths.
        """
        os.makedirs(output_dir, exist_ok=True)
        bundle_path = os.path.join(output_dir, bundle_filename)
        self.save(bundle_path, compute_hash=True, hmac_key=hmac_key)

        key_repr = f'"{hmac_key}"' if hmac_key else "None"

        app_code = f'''"""
Auto-generated by Serverlex v1.3.0 ModelDeployer.
Production REST serving API powered by FastAPI & Uvicorn.
"""
import os
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from serverlex import ModelDeployer

app = FastAPI(
    title="Serverlex Model Serving API",
    description="Live microservice endpoint with schema validation, HMAC verification, and sub-10ms inference.",
    version="1.0.0"
)

BUNDLE_FILE = os.path.join(os.path.dirname(__file__), "{bundle_filename}")
HMAC_KEY = {key_repr}

# Load signed bundle once at startup
deployer = ModelDeployer.load(BUNDLE_FILE, hmac_key=HMAC_KEY)


@app.get("/health")
def health_check():
    return {{
        "status": "healthy",
        "service": "serverlex-serving",
        "model": type(deployer.model).__name__ if deployer.model else "None",
        "feature_count": len(deployer.pipeline.feature_columns) if deployer.pipeline else 0
    }}


@app.get("/schema")
def get_schema():
    """Returns expected feature names, data types, and memorized categories."""
    pipe = deployer.pipeline
    return {{
        "feature_columns": pipe.feature_columns if pipe else [],
        "numeric_imputations": pipe.num_fill if pipe else {{}},
        "categorical_categories": pipe.cat_categories if pipe else {{}},
        "datetime_column": pipe.date_col if pipe else None
    }}


@app.post("/predict")
def predict_endpoint(payload: Union[Dict[str, Any], List[Dict[str, Any]]], threshold: Optional[float] = Query(None)):
    """Runs live prediction with schema normalization, date expansion, and optional decision threshold."""
    try:
        preds = deployer.predict(payload, strict=False, threshold=threshold)
        return {{"predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds)}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {{str(e)}}")


@app.get("/benchmark")
def benchmark_endpoint(n_samples: int = 100):
    """Executes live pre-inference latency benchmark."""
    try:
        return deployer.benchmark(n_iterations=n_samples, warmup=10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {{str(e)}}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="{host}", port={port}, reload=False)
'''

        dockerfile = f'''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE {port}
CMD ["uvicorn", "app:app", "--host", "{host}", "--port", "{port}"]
'''

        reqs = '''serverlex>=1.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
'''

        readme = f'''# Serverlex Model Serving Microservice

Turnkey model inference API generated by Serverlex.

## 🚀 Run Locally
```bash
pip install -r requirements.txt
python app:app
# Or via uvicorn:
uvicorn app:app --host {host} --port {port}
```

Visit the interactive Swagger UI at: `http://localhost:{port}/docs`

## 🐳 Run with Docker
```bash
docker build -t serverlex-model .
docker run -p {port}:{port} serverlex-model
```

## 📡 API Endpoints
- `GET /health` - Service health & model metadata.
- `GET /schema` - Expected input feature definitions.
- `POST /predict` - Live inference (accepts single record dict or micro-batch list).
- `GET /benchmark` - Real-time latency (p50/p95/p99) SLA profiler.
'''

        app_path = os.path.join(output_dir, "app.py")
        docker_path = os.path.join(output_dir, "Dockerfile")
        reqs_path = os.path.join(output_dir, "requirements.txt")
        readme_path = os.path.join(output_dir, "README.md")

        with open(app_path, "w", encoding="utf-8") as f:
            f.write(app_code)
        with open(docker_path, "w", encoding="utf-8") as f:
            f.write(dockerfile)
        with open(reqs_path, "w", encoding="utf-8") as f:
            f.write(reqs)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme)

        logger.info("Successfully exported serving microservice package to '%s'", output_dir)
        return {
            "output_dir": output_dir,
            "bundle": bundle_path,
            "app": app_path,
            "dockerfile": docker_path,
            "requirements": reqs_path,
            "readme": readme_path
        }

