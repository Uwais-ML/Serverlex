# Serverlex Technical Manual & Concept Documentation

This document provides a comprehensive deep-dive into each module in the **Serverlex** library. Each section details the underlying mathematical/statistical concepts, why the module is critical in production, complete syntax examples, and realistic terminal/logged outputs.

---

## Table of Contents
1. [`DataCheck.py` — Pre-Training Quality Guardrails (`DataGuard`)](#1-datacheckpy--dataguard)
2. [`Clean.py` — Data Sanitization Engine (`Clean`)](#2-cleanpy--clean)
3. [`Visualization.py` — Visual EDA & Verification (`Visualize`)](#3-visualizationpy--visualize)
4. [`Pipeline.py` — Leakage-Free Preprocessing & Benchmarks (`MLPipeline`)](#4-pipelinepy--mlpipeline)
5. [`FeatureSelector.py` — Mutual Information Pruning (`FeatureSelector`)](#5-featureselectorpy--featureselector)
6. [`ModelHyperparameter.py` — Nested & Non-Nested Tuning (`ModelHyperparameter`)](#6-modelhyperparameterpy--modelhyperparameter)
7. [`EnsembleBuilder.py` — Automated Stacking & Voting (`EnsembleBuilder`)](#7-ensemblebuilderpy--ensemblebuilder)
8. [`Evaluation.py` — Deep Diagnostics & Performance Curves (`ModelEvaluator`)](#8-evaluationpy--modelevaluator)
9. [`Explainability.py` — Feature Importance Analysis (`ModelExplainer`)](#9-explainabilitypy--modelexplainer)
10. [`ModelDeployer.py` — Production Bundling & Serving (`ModelDeployer`)](#10-modeldeployerpy--modeldeployer)
11. [`DriftDetector.py` — Production Distribution Shift Alert (`DriftDetector`)](#11-driftdetectorpy--driftdetector)

---

## 1. `DataCheck.py` — `DataGuard`

### The Concept & Problem
Real-world datasets frequently contain hidden pathologies that ruin models before training even starts:
1. **Target Leakage**: A feature that contains future information or is derived from the label (e.g., `refund_amount` predicting `return_status`). It yields 99.9% test accuracy but 0% in production.
2. **ID Memorization**: Features like `user_id` or `uuid` have near-100% uniqueness. Complex models (like deep trees) memorize these IDs rather than learning generalizable patterns.
3. **Class Imbalance**: In fraud or churn datasets, 95% of rows are class 0. A naive model that predicts 0 for everything gets 95% accuracy while being completely useless.

### API Reference
- `DataGuard(df)`: Initializes with a pandas DataFrame.
- `check_id_columns()`: Detects columns with $>95\%$ unique ratio or ID naming patterns.
- `check_target_leakage(target, threshold=0.95)`: Detects features with $|r| \ge \text{threshold}$ against target.
- `check_class_imbalance(target)`: Flags targets where dominant class $>70\%$.
- `check_cardinality(threshold=30)`: Identifies categories with $>30$ unique levels.
- `run_audit(target=None)`: Executes all checks and returns a summary DataFrame.

### Syntax
```python
from serverlex import DataGuard

guard = DataGuard(df)
audit_report = guard.run_audit(target="churn")
```

### Expected Output
```
2026-10-16 12:04:52 [INFO] DataGuard: Running data quality audit on 5000 rows, 12 columns
2026-10-16 12:04:52 [WARNING] DataGuard: Potential unique ID column detected: 'CustomerID' (uniqueness: 100.0%)
2026-10-16 12:04:52 [WARNING] DataGuard: Class imbalance in 'churn': dominant class is 82.4% (ratio 4.68:1)
2026-10-16 12:04:52 [INFO] DataGuard: Quality audit completed: 2 warnings/failures flagged
```

---

## 2. `Clean.py` — `Clean`

### The Concept & Problem
Messy categorical text strings and extreme outliers are the most common cause of degraded machine learning performance:
- **String Gaps & Case Fragmentation**: If categories are not standardized, `" New York "` and `"new york"` generate two distinct one-hot columns (`city_ New York ` and `city_newyork`), artificially fragmenting statistical support.
- **Outlier Distortion**: In standard scaling, $z = \frac{x - \mu}{\sigma}$. A single extreme outlier (\$50,000 when mean is \$500) blows up the standard deviation $\sigma$, squashing all normal data points into near-zero.
- **IQR Outlier Formulation**:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}$$

### API Reference
- `Clean(path)`: Initializes with a CSV path.
- `clean_column_names()`: Strips whitespace, lowercases, and replaces spaces with underscores.
- `clean_numeric_strings(columns)`: Strips `$` and `,` from numeric text and converts to float.
- `outliers_IQR(param='m')`: Identifies IQR bounds. `param='r'` removes rows; `param='m'` caps with mean; `param='median'` caps with median.
- `outliers_Z(param='m', threshold=3)`: Z-score outlier mitigation ($|z| > \text{threshold}$).
- `total_clean(...)`: Master pipeline executing all sanitization steps in verified order.

### Syntax
```python
from serverlex import Clean

cleaner = Clean("dirty_sales.csv")
clean_df = cleaner.total_clean(
    data_time=["Transaction Date", "%Y-%m-%d"],
    numeric_string_cols=["Product Price"],
    missing_param="m",
    outlier_method="IQR",
    outlier_param="m"
)
```

---

## 3. `Visualization.py` — `Visualize`

### The Concept & Problem
Data cleaning is often treated as a "black box" where developers cannot visually verify whether an outlier treatment or missing value imputation worked as intended. `Visualize` leverages Matplotlib's interactive mode (`plt.ion()`) to render live before-and-after plot updates as data moves through each stage.

### API Reference
- `plot_missing()`: Bar chart of missing values per column.
- `plot_distribution(column=None)`: Subplot histograms of numerical feature spreads.
- `plot_box(columns=None)`: Box plots to detect outliers before and after treatment.
- `plot_correlation()`: Heatmap annotated with Pearson correlation coefficients.
- `live_clean(pause_time=1.5)`: Runs cleaning while dynamically refreshing interactive figure windows.

### Syntax
```python
from serverlex import Visualize

vis = Visualize("raw_data.csv")
vis.plot_missing()
vis.plot_box()
vis.plot_correlation()

# Or watch the live step-by-step cleaning:
vis.live_clean(pause_time=2.0)
```

---

## 4. `Pipeline.py` — `MLPipeline`

### The Concept & Problem: Train-Test Leakage
In production machine learning, computing statistics (means, medians, scalers) across the entire dataset causes **Data Leakage**. Information from the test set leaks into training, inflating benchmark scores that immediately collapse in production.
Furthermore, if a categorical level appears in training but not in a single test record, standard one-hot encoding outputs mismatched column shapes, causing production crashes:
$$\text{Expected: } \mathbb{R}^{15} \neq \text{Received: } \mathbb{R}^{14} \implies \text{Crash}$$

`MLPipeline` solves this by strictly separating `fit()` and `transform()`:
- `fit(train_df)`: Calculates and memorizes all training means, medians, categories, and scalers on `self`.
- `transform(new_df)`: Transforms new data using the **exact memorized training rules**, and aligns feature columns with `.reindex(columns=self.feature_columns, fill_value=0)`.

### API Reference
- `fit(df, target=None, encode='onehot', scale='standard', data_time=None, numeric_string_cols=None)`: Learns rules from training data.
- `transform(df)`: Transforms incoming data using memorized training rules.
- `fit_transform(df, **kwargs)`: Fits and transforms in one step.
- `add_model(name, estimator)`: Registers custom models (XGBoost, LightGBM, CatBoost, KNN, etc.) into the benchmarking suite.
- `evaluate_baseline_models(X, y, cv=5)`: Auto-detects task (Classification vs. Regression) and runs cross-validation across all baseline and custom models.

### Syntax
```python
from serverlex import MLPipeline
from sklearn.neighbors import KNeighborsClassifier

pipeline = MLPipeline()

# 1. Fit on train, transform both
pipeline.fit(train_df, target="churn", data_time=["signup_date", "%Y-%m-%d"])
X_train, y_train = pipeline.transform(train_df)
X_test, y_test = pipeline.transform(test_df)

# 2. Add custom model & evaluate baseline leaderboard
pipeline.add_model("KNN", KNeighborsClassifier(n_neighbors=3))
leaderboard = pipeline.evaluate_baseline_models(X_train, y_train, cv=5)
```

### Expected Output
```
2026-10-16 12:05:44 [INFO] Pipeline: Fitting pipeline on 5000 rows (target=churn, encode=onehot, scale=standard)
2026-10-16 12:05:45 [INFO] Pipeline: Fit complete. Feature count: 12
2026-10-16 12:05:45 [INFO] Pipeline: Evaluating 5 baseline models (task=Classification, cv=5)
2026-10-16 12:05:46 [INFO] Pipeline: Baseline leaderboard:
              Model  Accuracy     F1
  Gradient Boosting    0.9942 0.9942
      Random Forest    0.9886 0.9886
                KNN    0.9620 0.9618
      Decision Tree    0.9501 0.9501
Logistic Regression    0.9042 0.9044
```

---

## 5. `FeatureSelector.py` — `FeatureSelector`

### The Concept & Problem
High-dimensional datasets (especially after one-hot encoding) suffer from the **Curse of Dimensionality**: noise features dilute predictive signals and dramatically increase training latency.
`FeatureSelector` applies a 3-stage pruning cascade:
1. **Variance Threshold**: Drops features with near-zero variance ($s^2 \le 0.01$).
2. **Collinearity Pruning**: Drops redundant features with pairwise correlation $>0.90$.
3. **Mutual Information Selection**:
   $$I(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x)p(y)} dx dy$$
   Unlike Pearson correlation (which only captures linear trends), Mutual Information captures arbitrary non-linear relationships with the target.

### API Reference
- `FeatureSelector(k=10, variance_threshold=0.01, corr_threshold=0.90)`: Configures selector.
- `fit(X, y)`: Calculates variance, collinearity, and mutual information scores.
- `transform(X)`: Filters input to the Top $K$ selected features.
- `get_ranked_scores()`: Returns a DataFrame ranking all features by mutual information.

### Syntax
```python
from serverlex import FeatureSelector

selector = FeatureSelector(k=8, corr_threshold=0.90)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
print(selector.get_ranked_scores())
```

---

## 6. `ModelHyperparameter.py` — `ModelHyperparameter`

### The Concept: Non-Nested vs. Nested Cross-Validation
- **Non-Nested CV (`tune_non_nested`)**: Runs standard `GridSearchCV`. Used to find the single best hyperparameter combination for **final production deployment**. However, because hyperparameters are chosen to maximize the score on those validation folds, the reported score can be slightly optimistic.
- **Nested Cross-Validation (`tune_nested_cv`)**:
  - **Outer Loop ($k_1$ folds)**: Evaluates overall generalization error on holdout test folds that hyperparameters were never fitted to.
  - **Inner Loop ($k_2$ folds)**: Runs `GridSearchCV` inside each outer training fold.
  - **Result**: A strictly **unbiased estimate of model generalization** required for scientific rigor and mission-critical production benchmarking.

### API Reference
- `get_default_param_grid(model_name)`: Returns pre-configured grids for popular algorithms.
- `tune_non_nested(estimator, param_grid, X, y, cv=5)`: Finds best deployment model.
- `tune_nested_cv(estimator, param_grid, X, y, outer_cv=5, inner_cv=3)`: Unbiased generalization benchmark.
- `compare_nested_vs_non_nested(...)`: Runs both side-by-side.

### Syntax
```python
from serverlex import ModelHyperparameter
from sklearn.ensemble import RandomForestClassifier

tuner = ModelHyperparameter()
param_grid = tuner.get_default_param_grid("random_forest")

res = tuner.tune_nested_cv(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    X=X_train,
    y=y_train,
    outer_cv=5,
    inner_cv=3
)
best_model = res["best_estimator"]
print("Unbiased Generalization Score:", res["nested_mean"])
```

---

## 7. `EnsembleBuilder.py` — `EnsembleBuilder`

### The Concept
Single machine learning algorithms have individual inductive biases (e.g., decision trees create axis-aligned step boundaries; logistic regression creates linear hyperplanes). Ensembling combines diverse algorithms:
- **Voting Ensemble**: Computes an average of predicted class probabilities:
  $$\hat{y} = \arg\max \frac{1}{M} \sum_{m=1}^{M} P_m(y = c \mid X)$$
- **Stacking Ensemble**: Uses out-of-fold predictions from $M$ base estimators as inputs to a **meta-learner** (e.g., Logistic Regression or Ridge), learning how to optimally weight base models.

### API Reference
- `EnsembleBuilder(task="classification")`: Sets task (`"classification"` or `"regression"`).
- `build_voting_ensemble(estimators, voting="soft")`: Creates a Voting ensemble.
- `build_stacking_ensemble(estimators, final_estimator=None, cv=5)`: Creates a Stacking ensemble with meta-learner.
- `evaluate_ensemble(ensemble, X, y, cv=5)`: Cross-validates ensemble performance.

### Syntax
```python
from serverlex import EnsembleBuilder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

builder = EnsembleBuilder(task="classification")
base_models = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("gb", GradientBoostingClassifier(random_state=42)),
    ("lr", LogisticRegression(max_iter=1000))
]

stack = builder.build_stacking_ensemble(base_models, cv=5)
scores = builder.evaluate_ensemble(stack, X_train, y_train, cv=5)
```

---

## 8. `Evaluation.py` — `ModelEvaluator`

### The Concept
A single accuracy or $R^2$ metric hides critical failure modes (e.g., a model predicting only class 0 on imbalanced data has high accuracy but zero recall). `ModelEvaluator` provides multi-dimensional diagnostics:
- **Classification**:
  - **Confusion Matrix**: Visualizes False Positives (Type I error) and False Negatives (Type II error).
  - **ROC Curve & AUC**: Measures ranking quality across all classification thresholds.
  - **Precision-Recall Curve**: Evaluates minority class detection under severe imbalance.
- **Regression**:
  - **Residuals vs. Predicted**: Checks homoscedasticity (constant variance of errors).
  - **Normality of Errors**: Verifies that residuals follow a Gaussian distribution without skew.

### API Reference
- `evaluate(model, X, y)`: Automatically detects task and executes full diagnostic suite.
- `plot_confusion_matrix(y_true, y_pred)`: Renders annotated confusion matrix.
- `plot_roc_curve(y_true, y_prob)`: Plots ROC curve and returns AUC.
- `plot_actual_vs_predicted(y_true, y_pred)`: Plots actual vs predicted scatter with $y=x$ reference.
- `plot_residuals(y_true, y_pred)`: Renders residuals scatter and error distribution.

### Syntax
```python
from serverlex import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.evaluate(model, X_test, y_test)
```

---

## 9. `Explainability.py` — `ModelExplainer`

### The Concept
Enterprise models cannot remain black boxes. Stakeholders and regulators require interpretability:
- **MDI (Mean Decrease in Impurity)**: Native tree importances measuring how much each feature decreases Gini impurity or variance across all splits.
- **Permutation Importance**: Model-agnostic measurement:
  $$I(f) = \text{Score}_{\text{original}} - \text{Score}_{\text{permuted}(f)}$$
  Shuffles a feature's values randomly. If model accuracy drops drastically, the feature is critical. If accuracy remains unchanged, the feature is useless.

### API Reference
- `ModelExplainer(model, feature_names=None)`: Initializes explainer.
- `get_feature_importances()`: Returns sorted DataFrame of feature importances or linear coefficients.
- `compute_permutation_importance(X, y)`: Computes model-agnostic permutation scores.
- `plot_feature_importance(top_n=10, save_path=None)`: Renders horizontal bar chart of top drivers.

### Syntax
```python
from serverlex import ModelExplainer

explainer = ModelExplainer(model, feature_names=X_train.columns)
print(explainer.get_feature_importances().head(5))
explainer.plot_feature_importance(top_n=10, title="Top Drivers of Churn")
```

---

## 10. `ModelDeployer.py` — `ModelDeployer`

### The Concept
In standard workflows, deploying a model requires maintaining separate preprocessing code in the production API. If the training pipeline used a mean of 34.2 and the API calculates 31.0, the model experiences silent accuracy degradation.
`ModelDeployer` serializes the fitted `MLPipeline` and trained `model` into a single artifact (`.joblib`). The production API accepts completely raw JSON inputs without requiring any preprocessing code.

### API Reference
- `ModelDeployer(pipeline=pipeline, model=model)`: Packages pipeline and model.
- `save(filepath="model_bundle.joblib")`: Serializes bundle.
- `load(filepath)`: Deserializes bundle.
- `predict(raw_input)`: Accepts raw dict, list of dicts, or uncleaned DataFrame and predicts in 1 line.
- `predict_proba(raw_input)`: Returns class probabilities for classifiers.

### Syntax
```python
from serverlex import ModelDeployer

# 1. Save artifact after training
deployer = ModelDeployer(pipeline=pipeline, model=best_model)
deployer.save("churn_deployer.joblib")

# 2. In production API (FastAPI / Flask):
prod = ModelDeployer.load("churn_deployer.joblib")
raw_customer = {"signup_date": "2026-04-01", "monthly_bill": "$89.50", "device": "laptop"}
prediction = prod.predict(raw_customer)
probabilities = prod.predict_proba(raw_customer)
```

---

## 11. `DriftDetector.py` — `DriftDetector`

### The Concept
Models degrade over time because real-world data distributions drift away from training distributions (**Covariate Shift**):
$$P_{\text{train}}(X) \neq P_{\text{production}}(X)$$
`DriftDetector` uses non-parametric hypothesis testing:
- **Two-Sample Kolmogorov-Smirnov Test (KS Test)**: Compares continuous empirical cumulative distribution functions (eCDFs):
  $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{current}}(x)|$$
  If $p\text{-value} < \alpha$ (default 0.05), the null hypothesis that both samples come from the same distribution is rejected.
- **Chi-Square Contingency Test**: Compares categorical frequency shifts across discrete bins.

### API Reference
- `DriftDetector(reference_data, alpha=0.05)`: Initializes with baseline training DataFrame.
- `detect_drift(current_data)`: Compares incoming production data against reference data and returns a detailed report.

### Syntax
```python
from serverlex import DriftDetector

detector = DriftDetector(reference_data=train_df, alpha=0.05)
drift_report = detector.detect_drift(current_data=production_batch_df)
print(drift_report)
```

### Expected Output
```
2026-10-16 12:52:29 [INFO] DriftDetector: Running drift detection across 12 features (alpha=0.050)
2026-10-16 12:52:29 [WARNING] DriftDetector: Data drift detected in 'monthly_bill' (KS=0.4120, p-value=0.00000)
2026-10-16 12:52:29 [INFO] DriftDetector: Drift analysis complete: 1/12 features drifted

     Feature        Type  Statistic  P_Value  Drift_Detected
monthly_bill   Numerical     0.4120   0.0000            True
      device Categorical     2.1402   0.3429           False
```
