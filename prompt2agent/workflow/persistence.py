"""Workflow persistence helpers."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from prompt2agent.config import WORKFLOWS_DIR, ensure_storage
from prompt2agent.utils.logging import get_logger
from prompt2agent.utils.text import slugify

logger = get_logger(__name__)
_INDEX_FILE = WORKFLOWS_DIR / "index.json"


def _load_index() -> List[Dict[str, Any]]:
    if not _INDEX_FILE.exists():
        return []
    return json.loads(_INDEX_FILE.read_text(encoding="utf-8"))


def _save_index(index: List[Dict[str, Any]]) -> None:
    _INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def save_workflow(workflow: Dict[str, Any], *, goal: str) -> Path:
    """Persist workflow JSON and update index."""
    ensure_storage()
    slug = slugify(goal)
    workflow_id = workflow.get("metadata", {}).get("id", uuid4().hex)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"workflow_{slug}_{workflow_id}_{timestamp}.json"
    path = WORKFLOWS_DIR / filename
    path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    logger.info("Saved workflow to %s", path)

    index = _load_index()
    index.append(
        {
            "file": path.name,
            "goal": goal,
            "workflow_id": workflow_id,
            "saved_at": timestamp,
        }
    )
    _save_index(index)
    return path


def load_workflow(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.debug("Loaded workflow from %s", path)
    return data


def list_workflows() -> List[Dict[str, Any]]:
    ensure_storage()
    return _load_index()
