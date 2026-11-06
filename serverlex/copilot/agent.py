"""
Serverlex AI Copilot: Built-in local diagnostic agent powered by fine-tuned 0.5B model.
Runs offline with zero external API calls or GPU required.
"""
import os
import sys
import logging
from typing import Optional, Union, Dict, Any

logger = logging.getLogger("ServerlexCopilot")

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

class ServerlexCopilot:
    """
    Offline local AI Copilot for Serverlex:
    - Auto-diagnoses DataGuard pre-flight quality and leakage logs.
    - Resolves ModelDeployer serving exceptions, HMAC mismatches, and schema violations.
    - Interprets DriftDetector PSI & KS distribution shifts.
    - Generates multi-step, leakage-safe Serverlex code workflows on demand.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_DIR
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Copilot model directory not found at '{self.model_path}'. "
                "Ensure serverlex-0.5b-merged weights are placed in serverlex/copilot/model/."
            )

        self.device = device or ("mps" if self._has_mps() else ("cuda" if self._has_cuda() else "cpu"))
        self._tokenizer = None
        self._model = None

    @staticmethod
    def _has_mps() -> bool:
        try:
            import torch
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            return False

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _lazy_load(self):
        """Loads the model into memory only when the first query is called."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "Copilot requires 'transformers' and 'torch'. Install them via: pip install torch transformers"
            ) from e

        logger.info("Loading Serverlex Copilot model on %s...", self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Use fp16 for MPS/CUDA, float32 for CPU
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device
        )
        self._model.eval()
        logger.info("Serverlex Copilot loaded successfully.")

    def ask(self, instruction: str, input_context: str = "", max_tokens: int = 350) -> str:
        """
        Sends an instruction and optional context to the local Copilot model.
        """
        self._lazy_load()
        import torch

        if input_context and str(input_context).strip():
            prompt = f"### Instruction:\n{instruction.strip()}\n\n### Input:\n{str(input_context).strip()}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.12,
                pad_token_id=self._tokenizer.pad_token_id
            )

        gen_tokens = outputs[0][len(inputs.input_ids[0]):]
        response = self._tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return response

    def diagnose_log(self, log_output: str) -> str:
        """Analyzes a Serverlex error or audit log and provides the diagnostic root cause + fix."""
        instruction = "Analyze this Serverlex log, diagnose the problem, and output the exact code fix."
        return self.ask(instruction, input_context=log_output)

    def generate_pipeline(self, target: str, description: str = "") -> str:
        """Generates a complete production Serverlex pipeline for a given target variable and requirements."""
        instruction = f"Write a complete, production-safe Serverlex pipeline to preprocess data, benchmark baseline algorithms, tune the best model, and save a signed deployment artifact for target '{target}'."
        context = description if description else f"Target column: '{target}'."
        return self.ask(instruction, input_context=context)


# Quick singleton helper for zero-boilerplate use
_copilot_instance: Optional[ServerlexCopilot] = None

def get_copilot(model_path: Optional[str] = None) -> ServerlexCopilot:
    global _copilot_instance
    if _copilot_instance is None:
        _copilot_instance = ServerlexCopilot(model_path=model_path)
    return _copilot_instance
