"""
Serverlex Auto-Copilot Hook (serverlex.copilot.hook)
Enables automatic error interception and Copilot triage guidance.
"""
import sys
import traceback
import logging

logger = logging.getLogger("ServerlexAutoCopilot")

_original_excepthook = sys.excepthook
_is_hook_active = False

def _copilot_excepthook(exc_type, exc_value, exc_traceback):
    # 1. First, print the normal Python traceback so standard debugging remains intact
    _original_excepthook(exc_type, exc_value, exc_traceback)

    # Ignore user interruptions (Ctrl+C)
    if issubclass(exc_type, KeyboardInterrupt):
        return

    # 2. Extract error information
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    error_summary = "".join(tb_lines[-3:]).strip()

    print("\n" + "=" * 70)
    print("🤖 [SERVERLEX COPILOT: AUTO-TRIAGE & DIAGNOSIS]")
    print("=" * 70)
    print("Analyzing failure with local 0.5B fine-tuned model...\n")

    try:
        from serverlex.copilot.agent import get_copilot
        copilot = get_copilot()
        diagnosis = copilot.diagnose_log(error_summary)
        print(diagnosis)
    except Exception as e:
        print(f"(Copilot auto-triage unavailable: {e})")
        print("💡 Quick Hint: Verify parameter names, target columns, and scaling modes in Serverlex documentation.")

    print("\n" + "=" * 70 + "\n")


def enable_auto_copilot():
    """
    Activates automatic exception interception.
    Whenever any script hits an unhandled error or exception, Serverlex Copilot
    automatically intercepts it, pinpoints the root cause, and outputs the exact code fix.
    """
    global _is_hook_active
    if not _is_hook_active:
        sys.excepthook = _copilot_excepthook
        _is_hook_active = True
        logger.info("Serverlex Auto-Copilot exception hook activated.")


def disable_auto_copilot():
    """Restores Python's default exception hook."""
    global _is_hook_active
    if _is_hook_active:
        sys.excepthook = _original_excepthook
        _is_hook_active = False
        logger.info("Serverlex Auto-Copilot exception hook deactivated.")
