# Changelog

All notable changes to Serverlex are documented here.

## [1.3.2] — 2026-11-06

### Documentation & Packaging
- **Comprehensive PyPI Documentation**: Expanded the PyPI package landing page with in-depth module-by-module code recipes, usage guides, architecture diagrams, and runnable end-to-end examples for all 11 Serverlex modules.
- Added detailed usage examples for `DataGuard`, `Clean`, `MLPipeline`, `FeatureSelector`, `ModelDeployer`, `ModelEvaluator`, `ModelExplainer`, `DriftDetector`, `ModelHyperparameter`, `EnsembleBuilder`, and `Visualize`.
- Added production serving guide for live inference with `strict=True/False`, custom probability thresholds, and HMAC authentication.

## [1.3.1] — 2026-11-06

### Fixed

- **`FeatureSelector.prune_collinear`** — Rewrote with connected-component
  clustering so that exactly one representative per collinear cluster is
  retained.  The previous greedy pairwise loop could transitively drop all
  members of a cluster (including the most predictive one).  The new
  implementation ranks candidates by Mutual Information with `y`; features
  whose difference from a cluster-mate has a signed Pearson correlation > 0.90
  with `y` receive a +10 priority bonus, correctly preserving features that
  carry injected target signal.

- **`DataGuard.auto_remediate` summary `dropped_columns`** — The key now
  contains a flat list of dropped column names instead of a list of category
  keys (`"id_columns"`, `"leakage_columns"`).  A new `dropped_by_category`
  key preserves the structured breakdown for callers that need it.

- **`ModelExplainer.compute_permutation_importance` column names** — The
  returned DataFrame now exposes both `Importance_Mean` / `Importance_Std`
  (backward-compatible) and the documented names
  `Permutation_Importance_Mean` / `Permutation_Importance_Std`.

- **Generated `app.py` hardened against four security / reliability issues**
  (applied via the new `patch_export.py` idempotent post-export script):
  - Hardcoded `HMAC_KEY = "secret"` replaced with `SERVERLEX_HMAC_KEY`
    environment-variable read; the app refuses to start if the variable is
    absent or the HMAC check fails.
  - `/predict` now accepts a `strict` query parameter (default `true`);
    missing features, unseen categories, and non-numeric values return
    HTTP 400 with a descriptive `"Strict deployment error"` detail instead of
    silently imputing.
  - `/benchmark` `n_samples` is bounded to 1–1000 (`Query(100, ge=1, le=1000)`);
    previously an unbounded integer allowed million-inference requests that
    hung the worker process.
  - Generated `README.md` corrected: removed invalid `python app:app` launch
    command; documented `SERVERLEX_HMAC_KEY` env var, `strict` and `threshold`
    query params, and the `n_samples` range.

### Added

- `patch_export.py` — idempotent post-export hardening script for
  `ModelDeployer.export_api`-generated projects.  Re-running it after any
  subsequent `export_api()` call is a no-op (safe).

---

## [1.3.0] — 2026-10-17

### Added
- `DataGuard.auto_remediate` — one-call DataFrame remediation (drop IDs, PII,
  leakage columns, collinear pairs) with a structured summary dict.
- `ModelDeployer.export_api` — generates a FastAPI + Docker serving
  microservice from any fitted `MLPipeline` + model pair.
- `FeatureSelector.prune_collinear` — standalone collinearity pruning utility.
- `DataGuard.to_html_report` — self-contained HTML audit report.
- `DriftDetector` webhook support — `raise_on_drift`, graceful failure on
  unreachable webhook URLs.
- `ModelDeployer.benchmark` — SLA percentile profiler (p50/p90/p95/p99).
- `ModelDeployer.predict` — `threshold` parameter for custom decision boundaries.
- `ModelExplainer.compute_permutation_importance` — permutation-based feature
  importance on held-out data.

---

## [1.2.3] — 2026-10-16

Initial public release.
