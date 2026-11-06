---
language:
- en
license: mit
tags:
- serverlex
- code-generation
- diagnostics
- machine-learning
- tabular
- lora
- auto-copilot
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: text-generation
library_name: transformers
---

# Serverlex Copilot 0.5B ⚡

**Serverlex Copilot 0.5B** is an offline, specialized AI diagnostic agent fine-tuned exclusively for [Serverlex](https://github.com/Uwais-ML/Serverlex)—the production-safe machine learning preprocessing, diagnostics, and serving platform.

At just **0.5 Billion parameters (~980MB in fp16, ~340MB in 4-bit)**, this model is lightweight enough to run locally on standard CPUs, Apple Silicon Macs, or edge environments with **zero external API calls or GPU requirements**.

---

## 📌 Capabilities

1. **Pre-Flight Data Quality Audit Triage**:
   - Analyzes `DataGuard` logs and audit reports.
   - Detects and explains ID memorization hazards, PII privacy concerns, and infinite value defects.
2. **Mathematical Target Leakage Mitigation**:
   - Identifies suspicious proxy features ($|r| \ge 0.90$) and writes exact code to drop or remediate them.
3. **Production Drift Interpretation**:
   - Interprets `DriftDetector` Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) statistics.
   - Prescribes retraining strategies and automated webhook alerting configurations.
4. **Hardened Serving & Cryptographic Verification**:
   - Diagnoses HMAC-SHA256 signature verification mismatches and tamper attempts in `ModelDeployer`.
   - Explains strict schema validation errors (unseen categories, missing features, invalid types).
5. **End-to-End Workflow Synthesis**:
   - Generates production-ready, leakage-free Serverlex code (cleaners, pipeline encoders, nested CV tuners, and containerized FastAPI exports).

---

## 🚀 Quickstart & Inference

### 1. Directly via Serverlex (Recommended)

```python
import serverlex

# Enable real-time automatic exception interception & guidance
serverlex.enable_auto_copilot()

# Or use the standalone Copilot agent
from serverlex import ServerlexCopilot

copilot = ServerlexCopilot()
log = "2026-11-06 10:39:12 [ERROR] DataGuard: Suspicious target leakage: feature post_op_charges has correlation 0.9842 with target total_hospital_cost"

print(copilot.diagnose_log(log))
```

### 2. Via Hugging Face `transformers`

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Uwais-ML-dev/serverlex-copilot-0.5b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = """### Instruction:
Analyze this Serverlex log and output the fix.

### Input:
2026-11-06 10:39:12 [ERROR] DataGuard: Suspicious target leakage: feature 'hospital_charges' has correlation 0.9842 with target 'total_hospital_cost'

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True))
```

---

## 🏋️ Training Details

- **Base Architecture**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Methodology**: LoRA (Low-Rank Adaptation) merged into standalone fp16 base weights
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Dataset**: 5,150 curated instructional pairs covering real Serverlex logs, multi-step recipes, and diagnostics.
- **Hardware**: Fine-tuned on a single NVIDIA T4 GPU in Google Colab.
- **Final Training Loss**: 0.1472

---

## 📄 License & Attribution

Released under the **MIT License**. Created by the Serverlex Team as part of the **Serverlex 2.0 Platform**.
