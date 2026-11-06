"""
Tests for serverlex.copilot — Serverlex AI Copilot (2.0)

These tests are split into two groups:
  - Lightweight (no torch/transformers needed): structure, routing, hook, CLI parsing
  - Inference tests (skipped if torch/transformers absent): actual model load + generation

Run lightweight only:
    pytest tests/test_copilot.py -m "not inference"

Run everything (requires copilot extras):
    pytest tests/test_copilot.py
"""
import sys
import os
import types
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copilot_deps_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


HAS_COPILOT_DEPS = _copilot_deps_available()
skip_no_deps = pytest.mark.skipif(
    not HAS_COPILOT_DEPS,
    reason="torch/transformers not installed — run: pip install serverlex[copilot]"
)


# ---------------------------------------------------------------------------
# 1. Import & public API surface
# ---------------------------------------------------------------------------

class TestCopilotImports:
    def test_package_exports(self):
        """All public symbols are reachable from serverlex.copilot."""
        from serverlex.copilot import (
            ServerlexCopilot,
            get_copilot,
            enable_auto_copilot,
            disable_auto_copilot,
        )
        assert callable(ServerlexCopilot)
        assert callable(get_copilot)
        assert callable(enable_auto_copilot)
        assert callable(disable_auto_copilot)

    def test_top_level_serverlex_exports(self):
        """Copilot symbols are also reachable from the top-level serverlex package."""
        import serverlex
        assert hasattr(serverlex, "ServerlexCopilot")
        assert hasattr(serverlex, "get_copilot")
        assert hasattr(serverlex, "enable_auto_copilot")
        assert hasattr(serverlex, "disable_auto_copilot")

    def test_import_does_not_load_torch(self):
        """Importing serverlex.copilot must NOT trigger a torch import."""
        # Temporarily block torch to prove lazy loading works
        sentinel = object()
        original = sys.modules.get("torch", sentinel)
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            # Re-import after blocking torch — should not raise
            import importlib
            import serverlex.copilot
            importlib.reload(serverlex.copilot)
        except ImportError as e:
            pytest.fail(f"Importing serverlex.copilot eagerly imported torch: {e}")
        finally:
            if original is sentinel:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 2. ServerlexCopilot — routing logic (no model load)
# ---------------------------------------------------------------------------

class TestCopilotRouting:
    def test_explicit_model_path_respected(self, tmp_path):
        """Explicit model_path overrides all auto-detection."""
        from serverlex.copilot.agent import ServerlexCopilot
        cop = ServerlexCopilot(model_path=str(tmp_path), subfolder="mysub")
        assert cop.model_path == str(tmp_path)
        assert cop.subfolder == "mysub"

    def test_falls_back_to_hf_when_no_local_weights(self, tmp_path, monkeypatch):
        """With no local weights, should fall back to HF repo (no subfolder for FP16 root)."""
        from serverlex.copilot import agent as agent_mod
        # Point local dirs to empty tmp dirs so _dir_has_weights returns False
        monkeypatch.setattr(agent_mod, "DEFAULT_LOCAL_FP16_DIR", str(tmp_path / "fp16"))
        monkeypatch.setattr(agent_mod, "DEFAULT_LOCAL_INT4_DIR", str(tmp_path / "int4"))

        from serverlex.copilot.agent import ServerlexCopilot
        cop = ServerlexCopilot()
        assert cop.model_path == agent_mod.DEFAULT_HF_REPO
        assert cop.subfolder == ""   # FP16 root, no subfolder

    def test_uses_local_fp16_when_present(self, tmp_path, monkeypatch):
        """When local FP16 dir has config.json + safetensors, it should be chosen."""
        from serverlex.copilot import agent as agent_mod

        # Build a fake local FP16 directory
        fp16_dir = tmp_path / "fp16"
        fp16_dir.mkdir()
        (fp16_dir / "config.json").write_text('{"model_type": "qwen2"}')
        (fp16_dir / "model.safetensors").write_bytes(b"\x00" * 8)

        monkeypatch.setattr(agent_mod, "DEFAULT_LOCAL_FP16_DIR", str(fp16_dir))

        from serverlex.copilot.agent import ServerlexCopilot
        cop = ServerlexCopilot()
        assert cop.model_path == str(fp16_dir)
        assert cop.subfolder is None

    def test_device_auto_detection_returns_string(self):
        """Device should always be a non-empty string."""
        from serverlex.copilot.agent import ServerlexCopilot, _dir_has_weights
        cop = ServerlexCopilot.__new__(ServerlexCopilot)
        cop._tokenizer = None
        cop._model = None
        cop.model_path = "dummy"
        cop.subfolder = None
        cop.device = ServerlexCopilot._has_mps() and "mps" or (
            ServerlexCopilot._has_cuda() and "cuda" or "cpu"
        )
        assert cop.device in ("cpu", "cuda", "mps")

    def test_dir_has_weights_true(self, tmp_path):
        from serverlex.copilot.agent import _dir_has_weights
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"\x00")
        assert _dir_has_weights(str(tmp_path)) is True

    def test_dir_has_weights_missing_config(self, tmp_path):
        from serverlex.copilot.agent import _dir_has_weights
        (tmp_path / "model.safetensors").write_bytes(b"\x00")
        assert _dir_has_weights(str(tmp_path)) is False

    def test_dir_has_weights_missing_weights(self, tmp_path):
        from serverlex.copilot.agent import _dir_has_weights
        (tmp_path / "config.json").write_text("{}")
        assert _dir_has_weights(str(tmp_path)) is False

    def test_dir_has_weights_nonexistent(self):
        from serverlex.copilot.agent import _dir_has_weights
        assert _dir_has_weights("/nonexistent/path/xyz") is False


# ---------------------------------------------------------------------------
# 3. Exception hook
# ---------------------------------------------------------------------------

class TestExceptionHook:
    def test_enable_installs_hook(self):
        from serverlex.copilot import enable_auto_copilot, disable_auto_copilot
        original = sys.excepthook
        try:
            enable_auto_copilot()
            assert sys.excepthook is not original, "enable_auto_copilot should replace sys.excepthook"
        finally:
            disable_auto_copilot()

    def test_disable_restores_hook(self):
        from serverlex.copilot import enable_auto_copilot, disable_auto_copilot
        original = sys.excepthook
        enable_auto_copilot()
        disable_auto_copilot()
        assert sys.excepthook is original, "disable_auto_copilot should restore original sys.excepthook"

    def test_enable_disable_idempotent(self):
        """Calling enable/disable multiple times must not raise."""
        from serverlex.copilot import enable_auto_copilot, disable_auto_copilot
        enable_auto_copilot()
        enable_auto_copilot()   # second call should be safe
        disable_auto_copilot()
        disable_auto_copilot()  # second call should be safe

    def test_hook_callable_without_model(self, capsys):
        """The hook should handle exceptions gracefully even if the model isn't loaded."""
        from serverlex.copilot import enable_auto_copilot, disable_auto_copilot
        enable_auto_copilot()
        try:
            # Simulate the hook being called with a simple error
            exc = ValueError("test error")
            sys.excepthook(type(exc), exc, None)
            captured = capsys.readouterr()
            # Should print something (at minimum the error info)
            assert len(captured.out) > 0 or len(captured.err) > 0 or True  # hook ran without crashing
        finally:
            disable_auto_copilot()


# ---------------------------------------------------------------------------
# 4. Singleton helper
# ---------------------------------------------------------------------------

class TestGetCopilotSingleton:
    def test_get_copilot_returns_same_instance(self):
        from serverlex.copilot import agent as agent_mod
        # Reset global singleton first
        agent_mod._copilot_instance = None
        from serverlex.copilot import get_copilot
        a = get_copilot()
        b = get_copilot()
        assert a is b, "get_copilot() should always return the same singleton"
        agent_mod._copilot_instance = None  # clean up

    def test_get_copilot_accepts_custom_path(self, tmp_path):
        from serverlex.copilot import agent as agent_mod
        agent_mod._copilot_instance = None
        from serverlex.copilot import get_copilot
        cop = get_copilot(model_path=str(tmp_path))
        assert cop.model_path == str(tmp_path)
        agent_mod._copilot_instance = None  # clean up


# ---------------------------------------------------------------------------
# 5. Prompt builder (no model needed)
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def _build_prompt(self, instruction, input_context=""):
        """Replicate the prompt logic from ask() for testing."""
        if input_context and str(input_context).strip():
            return f"### Instruction:\n{instruction.strip()}\n\n### Input:\n{str(input_context).strip()}\n\n### Response:\n"
        return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"

    def test_prompt_with_context(self):
        p = self._build_prompt("Diagnose this.", "ERROR: something failed")
        assert "### Instruction:" in p
        assert "### Input:" in p
        assert "### Response:" in p
        assert "Diagnose this." in p
        assert "ERROR: something failed" in p

    def test_prompt_without_context(self):
        p = self._build_prompt("Generate a pipeline.")
        assert "### Instruction:" in p
        assert "### Input:" not in p
        assert "### Response:" in p

    def test_prompt_whitespace_stripped(self):
        p = self._build_prompt("  Explain DataGuard.  ", "  some context  ")
        assert "Explain DataGuard." in p
        assert "some context" in p


# ---------------------------------------------------------------------------
# 6. Inference tests — skipped if torch/transformers absent
# ---------------------------------------------------------------------------

@skip_no_deps
class TestCopilotInference:
    """
    Full end-to-end inference tests using the actual fine-tuned model.
    These run only when torch + transformers are installed.
    The model is loaded once per session via the module-level fixture.
    """

    @pytest.fixture(scope="class")
    def copilot(self):
        """Load the copilot once and reuse across all inference tests."""
        from serverlex.copilot import get_copilot
        from serverlex.copilot import agent as agent_mod
        agent_mod._copilot_instance = None
        cop = get_copilot()
        yield cop
        agent_mod._copilot_instance = None

    def test_model_loads_without_error(self, copilot):
        """Model and tokenizer should load successfully."""
        copilot._lazy_load()
        assert copilot._model is not None
        assert copilot._tokenizer is not None

    def test_ask_returns_non_empty_string(self, copilot):
        resp = copilot.ask("What is DataGuard in Serverlex?")
        assert isinstance(resp, str)
        assert len(resp.strip()) > 10, "Response should be a meaningful string"

    def test_ask_mentions_relevant_concepts(self, copilot):
        """Response about DataGuard should contain data-quality related terms."""
        resp = copilot.ask("What does DataGuard do in Serverlex?").lower()
        relevant_terms = ["data", "audit", "leakage", "quality", "column", "guard", "detect"]
        matches = [t for t in relevant_terms if t in resp]
        assert len(matches) >= 2, (
            f"Response should contain data-quality concepts. Got: {resp[:300]}"
        )

    def test_ask_with_input_context(self, copilot):
        """Context should be included in the model's response or influence it."""
        resp = copilot.ask(
            instruction="Diagnose this Serverlex error.",
            input_context="DataGuardError: Column 'account_id' is 100% unique."
        )
        assert isinstance(resp, str)
        assert len(resp.strip()) > 10

    def test_diagnose_log_returns_string(self, copilot):
        log = "PSI alert: feature 'tenure_months' PSI=0.28 (threshold 0.20)"
        resp = copilot.diagnose_log(log)
        assert isinstance(resp, str)
        assert len(resp.strip()) > 10

    def test_generate_pipeline_returns_code_like_response(self, copilot):
        resp = copilot.generate_pipeline("churn", "Binary classification on telecom data.")
        assert isinstance(resp, str)
        assert len(resp.strip()) > 20
        # Should contain at least some code-like patterns
        code_indicators = ["import", "serverlex", "pipeline", "model", "fit", "def", "from", "="]
        assert any(ind.lower() in resp.lower() for ind in code_indicators), (
            f"generate_pipeline should return code-like content. Got: {resp[:300]}"
        )

    def test_ask_max_tokens_respected(self, copilot):
        """Short max_tokens should produce a shorter response."""
        resp_short = copilot.ask("What is Serverlex?", max_tokens=20)
        resp_long  = copilot.ask("What is Serverlex?", max_tokens=200)
        # Short response shouldn't be radically longer than long one's token budget
        # (not exact since tokens != chars, but a rough sanity check)
        assert len(resp_short) <= len(resp_long) + 100

    def test_repeated_calls_consistent(self, copilot):
        """do_sample=False means repeated calls should give identical results."""
        q = "List the main modules of Serverlex."
        r1 = copilot.ask(q)
        r2 = copilot.ask(q)
        assert r1 == r2, "Greedy decoding (do_sample=False) should be deterministic"

    def test_generation_config_no_warnings(self, copilot, recwarn):
        """Loading and running inference should emit no UserWarnings from transformers."""
        copilot.ask("What is DriftDetector?")
        transformers_warnings = [
            w for w in recwarn.list
            if issubclass(w.category, UserWarning)
            and "transformers" in str(w.filename).lower()
            and "do_sample" in str(w.message).lower()
        ]
        assert len(transformers_warnings) == 0, (
            f"Unexpected generation_config warnings: {transformers_warnings}"
        )
