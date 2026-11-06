# Welcome to Serverlex ⚡

**Serverlex** is an end-to-end Python library that takes raw, dirty data and transforms it into model-ready features, baseline benchmarks, tuned algorithms, and live production endpoints.

---

## 🌟 Why Serverlex?

Most real-world machine learning failures don't happen because of poor model architectures; they happen because of **silent data engineering defects**:

1. **Target Leakage**: Features collected concurrently with or after the target event inflate validation metrics ($r \ge 0.90$).
2. **Train/Serve Skew**: Scaling factors, imputation medians, or categorical vocabularies drift between offline training and production serving.
3. **ID Column Memorization**: Identifiers like `CustomerID` are mistakenly treated as categorical features, exploding one-hot dimensions and causing extreme memorization.
4. **Collinearity Transitive Wipes**: Greedy pairwise collinearity dropping deletes entire clusters of predictive variables.
5. **Untrusted Model Serialization**: Insecure unpickling of `.joblib` or `.pkl` files without cryptographic authentication.

Serverlex eliminates these risks with mathematical guarantees, automated data auditing, and cryptographically signed microservices.

---

## 🚀 Key Features

| Capability | Module | Benefit |
|---|---|---|
| **Pre-Flight Health Audit** | `DataGuard` | Detects IDs, target leakage, PII, and infinite values before fitting. |
| **Outlier & String Cleaning** | `Clean` | Handles dirty currency strings (`$1,200.50`) and clamps IQR outliers safely. |
| **Leak-Free Pipeline** | `MLPipeline` | Freezes training parameters so test and live inference never skew. |
| **Collinear Cluster Pruning** | `FeatureSelector` | Preserves the most predictive feature in a collinear cluster using Mutual Information. |
| **Turnkey Serving Export** | `ModelDeployer` | Exports a complete FastAPI service and Dockerfile in 1 line. |
| **Cryptographic Signatures** | `ModelDeployer` | HMAC-SHA256 bundle verification prevents loading tampered artifacts. |
| **Permutation Importance** | `ModelExplainer` | Evaluates feature power on held-out test data without Gini tree bias. |
| **Real-Time Drift Alerts** | `DriftDetector` | Fires Slack/PagerDuty webhooks when production PSI exceeds thresholds. |

---

## 📦 Installation

```bash
pip install serverlex
```
