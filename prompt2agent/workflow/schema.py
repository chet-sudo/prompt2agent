"""Workflow schema and validation utilities."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

from prompt2agent.utils.logging import get_logger

logger = get_logger(__name__)

WORKFLOW_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Prompt2AgentWorkflow",
    "type": "object",
    "required": ["metadata", "agents", "execution"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["id", "goal", "created_at"],
            "properties": {
                "id": {"type": "string"},
                "goal": {"type": "string"},
                "description": {"type": "string"},
                "created_at": {"type": "string"},
            },
        },
        "agents": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["id", "name", "instructions"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "instructions": {"type": "string"},
                    "model": {"type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "memory": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
            "default": [],
        },
        "memory": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "config": {"type": "object"},
                },
            },
            "default": [],
        },
        "connections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "condition": {"type": "string"},
                },
            },
            "default": [],
        },
        "execution": {
            "type": "object",
            "required": ["mode"],
            "properties": {
                "mode": {"type": "string", "enum": ["sequential", "peer" ]},
                "max_iterations": {"type": "integer", "minimum": 1, "default": 1},
                "retry_limit": {"type": "integer", "minimum": 0, "default": 0},
            },
        },
    },
}


class WorkflowValidationError(Exception):
    """Raised when workflow validation fails."""


def _validate_agent_ids(agents: Iterable[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for agent in agents:
        agent_id = agent.get("id")
        if not agent_id:
            raise WorkflowValidationError("Agent is missing an 'id'.")
        if agent_id in ids:
            raise WorkflowValidationError(f"Duplicate agent id '{agent_id}'.")
        ids.append(agent_id)
    return ids


def validate_workflow(workflow: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate workflow dictionary against the schema subset."""
    errors: List[str] = []
    try:
        json.dumps(workflow)
    except TypeError as err:
        errors.append(f"Workflow is not JSON serializable: {err}")
        return False, errors

    for field in ("metadata", "agents", "execution"):
        if field not in workflow:
            errors.append(f"Missing required field '{field}'.")

    metadata = workflow.get("metadata", {})
    for field in ("id", "goal", "created_at"):
        if not metadata.get(field):
            errors.append(f"metadata.{field} is required")

    agents = workflow.get("agents", [])
    if not isinstance(agents, list) or len(agents) < 2:
        errors.append("At least two agent definitions are required")
    else:
        try:
            _validate_agent_ids(agents)
        except WorkflowValidationError as err:
            errors.append(str(err))

    execution = workflow.get("execution", {})
    mode = execution.get("mode")
    if mode not in {"sequential", "peer"}:
        errors.append("execution.mode must be 'sequential' or 'peer'")

    return len(errors) == 0, errors
