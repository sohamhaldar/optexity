"""Records the steps that needed the LLM, so a later pass can correct them.

Inert unless OPTEXITY_FALLBACK_LOG is set.

A row is assembled from two places, because neither knows the whole story:
command_based_action_with_retry has the command that failed and why, and
log_interacted_locator has the locator the fallback settled on.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_pending: dict[int, dict] = {}


def log_path() -> str | None:
    return os.environ.get("OPTEXITY_FALLBACK_LOG") or None


def note_failure(step: int, command: str | None, error: str | None) -> None:
    if log_path():
        _pending[step] = {"step": step, "failed_command": command, "error": str(error or "")[:2000]}


def note_fallback(step: int, best_candidate: str | None) -> None:
    """Write the row. Called once the fallback has acted, which is the only
    point the correction is known."""
    path = log_path()
    if not path:
        return
    try:
        row = _pending.pop(step, {"step": step, "failed_command": None, "error": ""})
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row | {"best_candidate": best_candidate}) + "\n")
    except Exception as e:
        logger.warning(f"[fallback_log] step {step}: {type(e).__name__}: {e}")
