"""
Versioned Prompt Registry for Agentic-Graph-RAG.

All agent/service prompts are centralized here. Prompt versions live in
separate modules (v1.py, v2.py, …). The active version is selected by:
    1. Environment variable  AGRAG_PROMPT_VERSION  (e.g. "v1", "v2")
    2. Falls back to DEFAULT_VERSION below

Usage:
    from config.prompts import get_prompt, PROMPT_VERSION

    synthesis_prompt = get_prompt("SYNTHESIS_PROMPT")
    corrective_prompt = get_prompt("CORRECTIVE_PROMPT")

Adding a new version:
    1. Copy v1.py → v2.py
    2. Edit prompts as needed
    3. Set  AGRAG_PROMPT_VERSION=v2  in your environment / .env
"""
import importlib
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_VERSION = "v1"


def _active_version() -> str:
    return os.environ.get("AGRAG_PROMPT_VERSION", DEFAULT_VERSION).strip()


def _load_version_module(version: str):
    """Dynamically import config.prompts.<version>."""
    module_name = f"config.prompts.{version}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(
            f"Prompt version '{version}' not found. "
            f"Expected module: {module_name}. "
            f"Available versions should be config/prompts/v1.py, v2.py, etc."
        )


# Cache the loaded module for the process lifetime
_cached_module = None
_cached_version = None


def _get_module():
    global _cached_module, _cached_version
    version = _active_version()
    if _cached_module is None or _cached_version != version:
        _cached_module = _load_version_module(version)
        _cached_version = version
        logger.info(f"Loaded prompt version: {version}")
    return _cached_module


def get_prompt(key: str, *, version: Optional[str] = None) -> str:
    """
    Retrieve a prompt constant by name.

    Args:
        key: Prompt constant name, e.g. "SYNTHESIS_PROMPT", "CORRECTIVE_PROMPT"
        version: Override the env-var version for this call (useful for A/B tests)

    Returns:
        The prompt string.

    Raises:
        AttributeError: if the key doesn't exist in the version module.
    """
    if version:
        mod = _load_version_module(version)
    else:
        mod = _get_module()
    try:
        return getattr(mod, key)
    except AttributeError:
        available = [k for k in dir(mod) if k.isupper() and not k.startswith("_")]
        raise AttributeError(
            f"Prompt '{key}' not found in version '{version or _active_version()}'. "
            f"Available: {available}"
        )


def list_prompts(*, version: Optional[str] = None) -> list:
    """List all prompt constant names in the active (or specified) version."""
    mod = _load_version_module(version) if version else _get_module()
    return sorted(k for k in dir(mod) if k.isupper() and not k.startswith("_"))


def reload_prompts() -> None:
    """Force-reload the active prompt module (useful after hot-editing v1.py)."""
    global _cached_module, _cached_version
    _cached_module = None
    _cached_version = None
    _get_module()


# Convenience: expose the active version string as a callable
def get_prompt_version() -> str:
    """Return the active prompt version string."""
    return _active_version()


# For backwards-compat callers that do `from config.prompts import PROMPT_VERSION`,
# resolve eagerly at import time.
PROMPT_VERSION: str = _active_version()
