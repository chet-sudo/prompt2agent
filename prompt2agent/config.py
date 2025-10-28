"""Project configuration utilities."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = BASE_DIR / "workflows"
RUNS_DIR = BASE_DIR / "runs"
DEFAULT_MODEL = os.environ.get("PROMPT2AGENT_DEFAULT_MODEL", "openrouter/auto")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


def ensure_storage() -> None:
    """Ensure the storage directories exist."""
    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
