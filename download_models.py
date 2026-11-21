#!/usr/bin/env python3
"""
download_models.py
==================
Restores the fine-tuned Serverlex Copilot model from Hugging Face.

Usage:
    pip install huggingface_hub
    python download_models.py
"""

import os
import sys

HF_REPO = "Uwais-ML-dev/serverlex-copilot-0.5b"
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serverlex", "copilot", "model")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ 'huggingface_hub' is required. Run: pip install huggingface_hub")
    sys.exit(1)

def main():
    print(f"Downloading Serverlex Copilot model from {HF_REPO} to {TARGET_DIR}...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=TARGET_DIR,
        resume_download=True,
    )
    print("✔ Model successfully restored.")

if __name__ == "__main__":
    main()
