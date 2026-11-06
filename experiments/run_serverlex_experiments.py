import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import logging
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from serverlex import (
    DataGuard,
    Clean,
    Visualize,
    MLPipeline,
    FeatureSelector,
    ModelHyperparameter,
    EnsembleBuilder,
    ModelEvaluator,
    ModelExplainer,
    ModelDeployer,
    DriftDetector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ServerlexBenchmark")

# Setup output folders
OUTPUT_BASE = os.path.abspath("experiments/outputs")
OUT_MESSY = os.path.join(OUTPUT_BASE, "messy_cancer_lab")
OUT_NETFLIX = os.path.join(OUTPUT_BASE, "netflix_titles")
os.makedirs(OUT_MESSY, exist_ok=True)
os.makedirs(OUT_NETFLIX, exist_ok=True)


def run_messy_cancer_lab():
    print("\n" + "=" * 80)
    print(">>> 🔬 [DATASET 1] THOROUGH SERVERLEX RUN: MESSY CANCER LAB DATASET")
    print("=" * 80 + "\n")

    messy_path = "/Users/apple/.cache/kagglehub/datasets/eustusmurea/labtest-dataset/versions/1/messy_cancer_lab_dataset.csv"
    raw_df = pd.read_csv(messy_path)
    print(f"Loaded Raw Messy Dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(f"Columns: {raw_df.columns.tolist()}")

    # Inspect initial dirty characteristics
    print("\n[Initial Messy Profiling]:")
    print(f" - Missing values per column:\n{raw_df.isnull().sum()[raw_df.isnull().sum() > 0]}")
    print(f" - Dirty price sample: {raw_df['price'].dropna().head(6).tolist()}")
    print(f" - Inconsistent service cases: {raw_df['service'].value_counts(dropna=False).to_dict()}")
    print(f" - Inconsistent test_category cases: {raw_df['test_category'].value_counts().head(6).to_dict()}")

    # 1. DATAGUARD AUDIT
    print("\n--- STAGE 1: DataGuard Pre-Flight Safety & Quality Audit ---")
    guard = DataGuard(raw_df, target="service")
    audit_report = guard.run_audit(collinearity_threshold=0.85)
    print(audit_report.to_string(index=False))

    html_report_path = os.path.join(OUT_MESSY, "cancer_dataguard_audit.html")
    guard.to_html_report(html_report_path, title="Messy Cancer Lab Pre-Flight Audit")
    print(f"Saved DataGuard HTML Health Report to: {html_report_path}")

    # 2. LOW-LEVEL DATA SANITIZATION (Clean)
    print("\n--- STAGE 2: Low-Level Sanitization with Clean ---")
    cleaner = Clean(raw_df.copy())
    cleaner.clean_column_names()
    cleaner.drop_embedded_headers()
    cleaner.clean_numeric_strings(columns=["price"])
    cleaner.Clean_string()  # strips whitespace, lowercases all strings, normalizes categories
    
    # Check price conversion
    print(f"Cleaned price summary:\n{cleaner.data['price'].describe()}")
    
    # Outlier clipping on price using IQR
    cleaner.outliers_IQR(param='median', exclude=['service'])
    print(f"Price summary after IQR outlier mitigation:\n{cleaner.data['price'].describe()}")

    # 3. EDA & VISUALIZATIONS (Visualize)
    print("\n--- STAGE 3: Server-Safe Visualizations with Visualize ---")
    vis = Visualize(cleaner.data.copy())
    vis.plot_missing(save_path=os.path.join(OUT_MESSY, "missing_values.png"))
    vis.plot_distribution(column="price", save_path=os.path.join(OUT_MESSY, "price_distribution.png"))
    vis.plot_box(columns=["price"], save_path=os.path.join(OUT_MESSY, "price_box_plot.png"))
    print("Saved EDA plots (missing values, distribution, box plot) to output directory.")

    # 4. LEAKAGE-FREE PREPROCESSING & MODELING (MLPipeline)
    print("\n--- STAGE 4: Leakage-Free Preprocessing & Benchmarking (MLPipeline) ---")
    
    # Filter known service labels for supervised triage classification
    df_model = cleaner.data[cleaner.data["service"].notna()].copy()
    df_model = df_model[df_model["service"] != "unknown"].copy()
    print(f"Supervised triage dataset shape: {df_model.shape}")
    print(f"Class distribution:\n{df_model['service'].value_counts()}")

    # Split 80/20 train/test strictly before pipeline fitting
    train_df, test_df = train_test_split(df_model, test_size=0.20, random_state=42, stratify=df_model["service"])
    print(f"Train split: {train_df.shape}, Test split: {test_df.shape}")

    pipeline = MLPipeline()
    # High cardinality IDs and operational leak signout_date excluded; patient_name is PII
    pipeline.fit(
        train_df,
        target="service",
        encode="onehot",
        scale="standard",
        data_time=["creation_date"],
        id_cols=["order_id", "lab_number", "patient_name", "assigned_to_pathologist", "signout_date"],
        drop_id_cols=True,
        auto_remediate=True
    )

    X_train, y_train = pipeline.transform(train_df)
    X_test, y_test = pipeline.transform(test_df)
    print(f"Transformed X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Extracted features sample: {X_train.columns[:8].tolist()}")

    # Baseline Model Leaderboard
    print("\n>>> Baseline Model Benchmarking (3-Fold CV):")
    leaderboard = pipeline.evaluate_baseline_models(X_train, y_train, cv=3)
    print(leaderboard.to_string(index=False))

    # 5. COLLINEAR PRUNING & FEATURE SELECTION
    print("\n--- STAGE 5: Feature Selection & Collinear Pruning ---")
    pruned_X_train, dropped_collinear = FeatureSelector.prune_collinear(X_train, threshold=0.85, y=y_train)
    print(f"Collinear features dropped: {len(dropped_collinear)} features -> {dropped_collinear}")
    
    selector = FeatureSelector(k=min(25, pruned_X_train.shape[1]), corr_threshold=0.90)
    X_train_sel = selector.fit_transform(pruned_X_train, y_train)
    X_test_sel = selector.transform(X_test[pruned_X_train.columns])
    print(f"Selected top {X_train_sel.shape[1]} features via Mutual Information ranking.")
    print("Top 5 Ranked Features:")
    print(selector.get_ranked_scores().head(5).to_string(index=False))

    # 6. HYPERPARAMETER TUNING (ModelHyperparameter)
    print("\n--- STAGE 6: Hyperparameter Optimization (ModelHyperparameter) ---")
    tuner = ModelHyperparameter()
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [6, 12],
        "min_samples_split": [2, 5]
    }
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    tune_res = tuner.tune_non_nested(rf_base, param_grid, X_train, y_train, cv=3)
    best_rf = tune_res["best_estimator"]
    print(f"Optimal Random Forest Parameters: {tune_res['best_params']}")
    print(f"Best CV Score: {tune_res['best_score']:.4f}")

    # 7. ENSEMBLE BUILDER (EnsembleBuilder)
    print("\n--- STAGE 7: Multi-Model Stacking Ensemble ---")
    builder = EnsembleBuilder(task="classification")
    base_models = [
        ("rf", best_rf),
        ("gb", HistGradientBoostingClassifier(max_iter=50, max_depth=4, random_state=42)),
        ("lr", LogisticRegression(max_iter=500, random_state=42))
    ]
    stack_ensemble = builder.build_stacking_ensemble(base_models, cv=3)
    stack_ensemble.fit(X_train, y_train)
    ens_eval = builder.evaluate_ensemble(stack_ensemble, X_train, y_train, cv=3)
    acc = ens_eval.get('Accuracy', ens_eval.get('mean_score', 0.0))
    std = ens_eval.get('Std_Accuracy', ens_eval.get('std_score', 0.0))
    print(f"Stacking Ensemble CV Accuracy: {acc:.4f} (+/- {std:.4f})")

    # 8. DEEP TEST EVALUATION & EXPLAINABILITY
    print("\n--- STAGE 8: Test Set Evaluation & Model Explainability ---")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(best_rf, X_test, y_test)
    print(f"Held-Out Test Metrics:\n{json.dumps(metrics, indent=2)}")

    evaluator.plot_confusion_matrix(y_test, best_rf.predict(X_test), save_path=os.path.join(OUT_MESSY, "confusion_matrix.png"))

    explainer = ModelExplainer(best_rf, feature_names=X_train.columns)
    importances = explainer.get_feature_importances()
    print("\nTop 5 Permutation/Tree Feature Drivers:")
    print(importances.head(5).to_string(index=False))
    explainer.plot_feature_importance(top_n=10, save_path=os.path.join(OUT_MESSY, "feature_importance.png"))

    # 9. PRODUCTION DEPLOYMENT & SLA BENCHMARKING (ModelDeployer)
    print("\n--- STAGE 9: Cryptographic Bundle Deployment & SLA Profiling ---")
    bundle_path = os.path.join(OUT_MESSY, "cancer_triage_bundle.joblib")
    
    # We deploy the pipeline and model
    deployer = ModelDeployer(pipeline=pipeline, model=best_rf, metadata={"task": "cancer_service_triage", "author": "Antigravity Serverlex"})
    deployer.save(bundle_path, compute_hash=True)
    print(f"Saved Cryptographically Signed Bundle to: {bundle_path}")
    print(f"SHA-256 Sidecar created: {bundle_path}.sha256")

    # Verify reload integrity
    loaded_deployer = ModelDeployer.load(bundle_path, verify_hash=True)
    print("Cryptographic verification PASSED upon reload.")

    # SLA Profiling
    sla = loaded_deployer.benchmark(n_iterations=200, warmup=20)
    print(f"SLA Profile: p50={sla['p50_ms']:.2f}ms | p95={sla['p95_ms']:.2f}ms | p99={sla['p99_ms']:.2f}ms | Throughput={sla['throughput_req_per_sec']:.1f} req/s")

    # Raw Payload Inference Simulation (dirty raw inputs)
    raw_sample = {
        "order_id": "ORD9999",
        "lab_number": "LAB9999",
        "patient_name": "Jane Doe",
        "assigned_to_pathologist": "Dr. Smith",
        "facility": "Mercy Cancer Center",
        "test_category": "Cytology",
        "test": "Fine Needle Aspiration",
        "sub_category": "Biopsy",
        "receiving_centers": "Center A",
        "processing_centers": "Center B",
        "creation_date": "2021-09-15",
        "price": "ksh 4,800",
        "payment_method": "Insurance"
    }
    raw_pred = loaded_deployer.predict(raw_sample)
    raw_prob = loaded_deployer.predict_proba(raw_sample)
    print(f"\nLive Raw Payload Inference: Prediction = '{raw_pred[0]}'")
    print(f"Predicted Class Probabilities: {raw_prob[0]}")

    # 10. REAL-TIME PRODUCTION COVARIATE DRIFT MONITORING (DriftDetector)
    print("\n--- STAGE 10: Production Covariate Shift Monitoring (DriftDetector) ---")
    # Simulate a shifted production batch: price surge + altered facility distribution
    shifted_batch = train_df.sample(200, random_state=99).copy()
    shifted_batch["price"] = shifted_batch["price"] * 2.5  # 250% price inflation
    shifted_batch["facility"] = "Unknown Emerging Clinic"

    detector = DriftDetector(reference_data=train_df, alpha=0.01, psi_threshold=0.20)
    drift_report = detector.detect_drift(current_data=shifted_batch, raise_on_drift=False)
    print("Drift Detection Summary Table:")
    print(drift_report[["Feature", "PSI", "Drift_Detected", "Severity"]].to_string(index=False))

    return {
        "dataset": "Messy Cancer Lab",
        "raw_shape": raw_df.shape,
        "clean_shape": cleaner.data.shape,
        "cv_score": tune_res["best_score"],
        "test_metrics": metrics,
        "sla": sla
    }


def run_netflix_titles():
    print("\n" + "=" * 80)
    print(">>> 🎬 [DATASET 2] THOROUGH SERVERLEX RUN: NETFLIX TITLES DATASET")
    print("=" * 80 + "\n")

    netflix_path = "netflix_titles.csv"
    raw_df = pd.read_csv(netflix_path)
    print(f"Loaded Raw Netflix Dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
    print(f"Columns: {raw_df.columns.tolist()}")

    # 1. DATAGUARD AUDIT
    print("\n--- STAGE 1: DataGuard Pre-Flight Safety & Quality Audit ---")
    guard = DataGuard(raw_df, target="type")
    audit_report = guard.run_audit(collinearity_threshold=0.85)
    print(audit_report.to_string(index=False))

    html_report_path = os.path.join(OUT_NETFLIX, "netflix_dataguard_audit.html")
    guard.to_html_report(html_report_path, title="Netflix Titles Pre-Flight Health Audit")
    print(f"Saved DataGuard HTML Health Report to: {html_report_path}")

    # 2. LOW-LEVEL DATA SANITIZATION (Clean)
    print("\n--- STAGE 2: Low-Level Sanitization with Clean ---")
    cleaner = Clean(raw_df.copy())
    cleaner.clean_column_names()
    cleaner.drop_embedded_headers()
    cleaner.delete_duplicates()

    # In Netflix, 'country' and 'listed_in' have comma-separated lists.
    # Extract primary country and primary genre to avoid sparse 1500-column explosion:
    cleaner.data["primary_country"] = cleaner.data["country"].fillna("Unknown").astype(str).apply(lambda x: x.split(",")[0].strip())
    cleaner.data["primary_genre"] = cleaner.data["listed_in"].fillna("Unknown").astype(str).apply(lambda x: x.split(",")[0].strip())
    
    # Coerce duration into clean numeric value (e.g. 90 min -> 90.0, 2 Seasons -> 2.0)
    cleaner.clean_numeric_strings(columns=["duration"])
    cleaner.Clean_string()  # normalize casing and trim whitespace

    print(f"Sanitized Netflix DataFrame shape: {cleaner.data.shape}")
    print(f"Duration numeric stats:\n{cleaner.data['duration'].describe()}")
    print(f"Primary Country top 5:\n{cleaner.data['primary_country'].value_counts().head(5)}")
    print(f"Primary Genre top 5:\n{cleaner.data['primary_genre'].value_counts().head(5)}")

    # 3. EDA & VISUALIZATIONS (Visualize)
    print("\n--- STAGE 3: Server-Safe Visualizations with Visualize ---")
    vis = Visualize(cleaner.data.copy())
    vis.plot_missing(save_path=os.path.join(OUT_NETFLIX, "missing_values.png"))
    vis.plot_distribution(column="release_year", save_path=os.path.join(OUT_NETFLIX, "release_year_distribution.png"))
    vis.plot_box(columns=["duration"], save_path=os.path.join(OUT_NETFLIX, "duration_box_plot.png"))
    print("Saved EDA plots (missing values, release year, duration) to output directory.")

    # 4. LEAKAGE-FREE PREPROCESSING & MODELING (MLPipeline)
    print("\n--- STAGE 4: Leakage-Free Preprocessing & Benchmarking (MLPipeline) ---")
    df_model = cleaner.data.copy()
    
    # Target binary classification: Movie vs TV Show
    print(f"Target distribution:\n{df_model['type'].value_counts()}")

    train_df, test_df = train_test_split(df_model, test_size=0.20, random_state=42, stratify=df_model["type"])
    print(f"Train split: {train_df.shape}, Test split: {test_df.shape}")

    pipeline = MLPipeline()
    # High cardinality text fields excluded; date_added extracted as datetime features
    pipeline.fit(
        train_df,
        target="type",
        encode="onehot",
        scale="standard",
        data_time=["date_added"],
        id_cols=["show_id", "title", "director", "cast", "description", "country", "listed_in"],
        drop_id_cols=True,
        auto_remediate=True
    )

    X_train, y_train = pipeline.transform(train_df)
    X_test, y_test = pipeline.transform(test_df)
    print(f"Transformed X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Feature count: {X_train.shape[1]}")

    # Baseline Model Leaderboard
    print("\n>>> Baseline Model Benchmarking (3-Fold CV):")
    leaderboard = pipeline.evaluate_baseline_models(X_train, y_train, cv=3)
    print(leaderboard.to_string(index=False))

    # 5. COLLINEAR PRUNING & FEATURE SELECTION
    print("\n--- STAGE 5: Feature Selection & Collinear Pruning ---")
    pruned_X_train, dropped_collinear = FeatureSelector.prune_collinear(X_train, threshold=0.85, y=y_train)
    print(f"Collinear features dropped: {len(dropped_collinear)} features -> {dropped_collinear}")
    
    selector = FeatureSelector(k=min(25, pruned_X_train.shape[1]), corr_threshold=0.90)
    X_train_sel = selector.fit_transform(pruned_X_train, y_train)
    X_test_sel = selector.transform(X_test[pruned_X_train.columns])
    print(f"Selected top {X_train_sel.shape[1]} features via Mutual Information ranking.")
    print("Top 5 Ranked Features:")
    print(selector.get_ranked_scores().head(5).to_string(index=False))

    # 6. HYPERPARAMETER TUNING (ModelHyperparameter)
    print("\n--- STAGE 6: Hyperparameter Optimization (ModelHyperparameter) ---")
    tuner = ModelHyperparameter()
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [6, 12],
        "min_samples_split": [2, 5]
    }
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    tune_res = tuner.tune_non_nested(rf_base, param_grid, X_train, y_train, cv=3)
    best_rf = tune_res["best_estimator"]
    print(f"Optimal Random Forest Parameters: {tune_res['best_params']}")
    print(f"Best CV Score: {tune_res['best_score']:.4f}")

    # 7. ENSEMBLE BUILDER (EnsembleBuilder)
    print("\n--- STAGE 7: Voting Ensemble ---")
    builder = EnsembleBuilder(task="classification")
    base_models = [
        ("rf", best_rf),
        ("gb", HistGradientBoostingClassifier(max_iter=50, max_depth=4, random_state=42)),
        ("lr", LogisticRegression(max_iter=500, random_state=42))
    ]
    voting_ensemble = builder.build_voting_ensemble(base_models, voting="soft")
    voting_ensemble.fit(X_train, y_train)
    ens_eval = builder.evaluate_ensemble(voting_ensemble, X_train, y_train, cv=3)
    acc = ens_eval.get('Accuracy', ens_eval.get('mean_score', 0.0))
    std = ens_eval.get('Std_Accuracy', ens_eval.get('std_score', 0.0))
    print(f"Voting Ensemble CV Accuracy: {acc:.4f} (+/- {std:.4f})")

    # 8. DEEP TEST EVALUATION & EXPLAINABILITY
    print("\n--- STAGE 8: Test Set Evaluation & Model Explainability ---")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(best_rf, X_test, y_test)
    print(f"Held-Out Test Metrics:\n{json.dumps(metrics, indent=2)}")

    evaluator.plot_confusion_matrix(y_test, best_rf.predict(X_test), save_path=os.path.join(OUT_NETFLIX, "confusion_matrix.png"))

    explainer = ModelExplainer(best_rf, feature_names=X_train.columns)
    importances = explainer.get_feature_importances()
    print("\nTop 5 Permutation/Tree Feature Drivers:")
    print(importances.head(5).to_string(index=False))
    explainer.plot_feature_importance(top_n=10, save_path=os.path.join(OUT_NETFLIX, "feature_importance.png"))

    # 9. PRODUCTION DEPLOYMENT & SLA BENCHMARKING (ModelDeployer)
    print("\n--- STAGE 9: Cryptographic Bundle Deployment & SLA Profiling ---")
    bundle_path = os.path.join(OUT_NETFLIX, "netflix_classifier_bundle.joblib")
    
    deployer = ModelDeployer(pipeline=pipeline, model=best_rf, metadata={"task": "netflix_type_prediction", "author": "Antigravity Serverlex"})
    deployer.save(bundle_path, compute_hash=True)
    print(f"Saved Cryptographically Signed Bundle to: {bundle_path}")
    print(f"SHA-256 Sidecar created: {bundle_path}.sha256")

    # Verify reload integrity
    loaded_deployer = ModelDeployer.load(bundle_path, verify_hash=True)
    print("Cryptographic verification PASSED upon reload.")

    # SLA Profiling
    sla = loaded_deployer.benchmark(n_iterations=200, warmup=20)
    print(f"SLA Profile: p50={sla['p50_ms']:.2f}ms | p95={sla['p95_ms']:.2f}ms | p99={sla['p99_ms']:.2f}ms | Throughput={sla['throughput_req_per_sec']:.1f} req/s")

    # Raw Payload Inference Simulation (dirty raw inputs)
    raw_sample = {
        "show_id": "s99999",
        "title": "Cosmic Voyage",
        "director": "Christopher Nolan",
        "cast": "Matthew McConaughey, Anne Hathaway",
        "country": "United States, United Kingdom",
        "date_added": "October 14, 2023",
        "release_year": 2021,
        "rating": "PG-13",
        "duration": "169 min",
        "listed_in": "Sci-Fi & Fantasy, Dramas",
        "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."
    }
    raw_pred = loaded_deployer.predict(raw_sample)
    raw_prob = loaded_deployer.predict_proba(raw_sample)
    print(f"\nLive Raw Payload Inference: Prediction = '{raw_pred[0]}'")
    print(f"Predicted Class Probabilities: {raw_prob[0]}")

    # 10. REAL-TIME PRODUCTION COVARIATE DRIFT MONITORING (DriftDetector)
    print("\n--- STAGE 10: Production Covariate Shift Monitoring (DriftDetector) ---")
    # Simulate a shifted production batch: older vintage movies (e.g. 1970-1990)
    shifted_batch = train_df.sample(200, random_state=88).copy()
    shifted_batch["release_year"] = np.random.randint(1970, 1990, size=200)

    detector = DriftDetector(reference_data=train_df, alpha=0.01, psi_threshold=0.20)
    drift_report = detector.detect_drift(current_data=shifted_batch, raise_on_drift=False)
    print("Drift Detection Summary Table:")
    print(drift_report[["Feature", "PSI", "Drift_Detected", "Severity"]].to_string(index=False))

    return {
        "dataset": "Netflix Titles",
        "raw_shape": raw_df.shape,
        "clean_shape": cleaner.data.shape,
        "cv_score": tune_res["best_score"],
        "test_metrics": metrics,
        "sla": sla
    }


if __name__ == "__main__":
    t0 = time.time()
    cancer_res = run_messy_cancer_lab()
    netflix_res = run_netflix_titles()
    elapsed = time.time() - t0
    
    print("\n" + "#" * 80)
    print(f"### ALL SERVERLEX PIPELINES COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS ###")
    print("#" * 80)
