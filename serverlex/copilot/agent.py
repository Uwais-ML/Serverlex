"""
Serverlex AI Copilot: Built-in local diagnostic agent powered by fine-tuned 0.5B model.
Runs offline with zero external API calls or GPU required.

Model sources (in priority order):
  1. Local FP16 weights:  serverlex/copilot/model/           (~942 MB, fastest cold-start)
  2. Local INT4 weights:  serverlex/copilot/model_int4/       (495 MB, custom nibble-packed)
  3. Hugging Face Hub:    Uwais-ML-dev/serverlex-copilot-0.5b (auto-downloaded on first use)
"""
import os
import sys
import logging
from typing import Optional

logger = logging.getLogger("ServerlexCopilot")

# --- Path constants -----------------------------------------------------------
_HERE = os.path.dirname(__file__)
DEFAULT_LOCAL_FP16_DIR  = os.path.join(_HERE, "model")       # standard HF layout
DEFAULT_LOCAL_INT4_DIR  = os.path.join(_HERE, "model_int4")  # nibble-packed custom
DEFAULT_HF_REPO         = "Uwais-ML-dev/serverlex-copilot-0.5b"
# HF root has the full FP16 model (config.json + model.safetensors); int4/ subfolder
# uses a custom packing format and cannot be loaded by AutoModelForCausalLM directly.
DEFAULT_HF_SUBFOLDER    = ""   # load from repo root (standard FP16 weights)


def _dir_has_weights(path: str) -> bool:
    """Return True if *path* contains a loadable model checkpoint."""
    if not os.path.isdir(path):
        return False
    has_safe  = any(f.endswith(".safetensors") for f in os.listdir(path))
    has_bin   = any(f.endswith(".bin")         for f in os.listdir(path))
    has_cfg   = os.path.exists(os.path.join(path, "config.json"))
    return has_cfg and (has_safe or has_bin)


class ServerlexCopilot:
    """
    Offline-first local AI Copilot for Serverlex.

    Capabilities
    ------------
    - Auto-diagnoses DataGuard pre-flight quality and data-leakage logs.
    - Resolves ModelDeployer serving exceptions, HMAC mismatches, and schema violations.
    - Interprets DriftDetector PSI & KS distribution shifts.
    - Generates multi-step, leakage-safe Serverlex code workflows on demand.

    Model loading priority
    ----------------------
    1. ``model_path`` argument (explicit override)
    2. Local FP16 weights at ``serverlex/copilot/model/``  (942 MB)
    3. Auto-download from Hugging Face Hub on first use    (~942 MB, cached by HF)

    Parameters
    ----------
    model_path : str, optional
        Explicit local directory or HF repo-id.  Auto-detected when omitted.
    subfolder  : str, optional
        HF subfolder (only used when ``model_path`` is an HF repo-id).
    device     : str, optional
        ``"cpu"``, ``"cuda"``, or ``"mps"``.  Auto-detected when omitted.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        subfolder:  Optional[str] = None,
        device:     Optional[str] = None,
    ):
        if model_path:
            # Explicit override — trust the caller
            self.model_path = model_path
            self.subfolder  = subfolder or ""
        elif _dir_has_weights(DEFAULT_LOCAL_FP16_DIR):
            # Prefer the full FP16 local copy (standard HF layout, no custom loader)
            self.model_path = DEFAULT_LOCAL_FP16_DIR
            self.subfolder  = None
            logger.debug("Copilot: using local FP16 weights at %s", DEFAULT_LOCAL_FP16_DIR)
        else:
            # Fall back to auto-download from Hugging Face Hub
            self.model_path = DEFAULT_HF_REPO
            self.subfolder  = subfolder if subfolder is not None else DEFAULT_HF_SUBFOLDER
            logger.info(
                "Copilot: local weights not found — will download from HF (%s) on first use.",
                DEFAULT_HF_REPO,
            )

        self.device = device or (
            "mps"  if self._has_mps()  else
            "cuda" if self._has_cuda() else
            "cpu"
        )
        self._tokenizer = None
        self._model     = None

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _lazy_load(self):
        """Load model + tokenizer into memory on the first inference call."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "Serverlex Copilot requires 'transformers' and 'torch'.\n"
                "Install them with:  pip install serverlex[copilot]"
            ) from e

        logger.info("Loading Serverlex Copilot model on %s …", self.device)

        # --- Tokenizer ---
        tok_kwargs: dict = {"trust_remote_code": False}
        if self.subfolder:
            tok_kwargs["subfolder"] = self.subfolder

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, **tok_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # --- Model ---
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        model_kwargs: dict = {
            "torch_dtype":        dtype,
            "device_map":         self.device,
            "trust_remote_code":  False,
            "low_cpu_mem_usage":  True,
        }
        if self.subfolder:
            model_kwargs["subfolder"] = self.subfolder

        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self._model.eval()

        # Suppress generation_config warnings from training artefacts.
        # We always call generate() with explicit do_sample=False, so ignore
        # any temperature/top_p/top_k baked into the saved generation_config.
        try:
            self._model.generation_config = GenerationConfig(
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        except Exception:
            pass

        logger.info("Serverlex Copilot ready (device=%s).", self.device)

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
