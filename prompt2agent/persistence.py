"""Persistence helpers for workflow specifications."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from models import WorkflowSpec
from workflow import RunnableWorkflow, materialise_workflow

logger = logging.getLogger(__name__)

_STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"
_WORKFLOW_DIR = _STORAGE_ROOT / "workflows"
_SESSION_DIR = _STORAGE_ROOT / "sessions"


def save_workflow(spec: WorkflowSpec) -> str:
    """Persist the workflow specification to disk and return its identifier."""

    _WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _WORKFLOW_DIR / f"{spec.metadata.spec_id}.json"
    data = spec.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    logger.info("Saved workflow specification to %s", path)
    return spec.metadata.spec_id


def load_workflow(workflow_id: str) -> RunnableWorkflow:
    """Load a workflow specification and materialise it into runnable agents."""

    path = _WORKFLOW_DIR / f"{workflow_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No workflow specification stored at {path}")

    data = json.loads(path.read_text())
    spec = WorkflowSpec.model_validate(data)
    logger.info("Loaded workflow specification '%s'", spec.metadata.title)
    return materialise_workflow(spec)


def session_store_path(workflow_id: str) -> Path:
    """Return the SQLite path used for persistent session storage."""

    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSION_DIR / f"{workflow_id}.sqlite"


__all__ = ["load_workflow", "save_workflow", "session_store_path"]
