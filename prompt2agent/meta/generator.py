"""Workflow generation using the meta agent."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict

from prompt2agent.adapters.model_adapter import ModelAdapter
from prompt2agent.utils.logging import get_logger
from prompt2agent.utils.text import slugify
from prompt2agent.workflow.schema import WORKFLOW_JSON_SCHEMA, validate_workflow

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You generate JSON workflow definitions for a multi-agent orchestration system. "
    "Respond with JSON that follows this schema: {schema}. Do not include explanations."
)


async def _llm_generate(prompt: str, adapter: ModelAdapter) -> Dict[str, Any]:
    schema_str = json.dumps(WORKFLOW_JSON_SCHEMA)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(schema=schema_str)},
        {
            "role": "user",
            "content": (
                "Create a workflow that fulfills this goal. Provide tool/memory references "
                "only if necessary. Goal: "
                + prompt
            ),
        },
    ]
    response = await adapter.chat(messages)
    logger.debug("Meta-agent raw response: %s", response)
    try:
        data = json.loads(response)
        return data
    except json.JSONDecodeError as err:
        raise ValueError(f"Meta agent returned invalid JSON: {err}") from err


def _fallback_workflow(prompt: str) -> Dict[str, Any]:
    slug = slugify(prompt)
    now = datetime.utcnow().isoformat()
    workflow_id = f"fallback-{slug}"
    workflow = {
        "metadata": {
            "id": workflow_id,
            "goal": prompt,
            "description": "Fallback sequential workflow",
            "created_at": now,
        },
        "agents": [
            {
                "id": "planner",
                "name": "Planner",
                "instructions": (
                    "Analyze the goal and produce a clear plan with numbered steps."
                ),
                "model": None,
                "tools": [],
                "memory": ["planner_buffer"],
            },
            {
                "id": "executor",
                "name": "Executor",
                "instructions": (
                    "Using the plan, execute each step logically. Summarize the outcome."
                ),
                "model": None,
                "tools": ["web_search"],
                "memory": ["executor_buffer"],
            },
        ],
        "tools": [
            {
                "name": "web_search",
                "type": "web_search",
                "description": "Perform a general web search using a public API.",
            }
        ],
        "memory": [
            {"name": "planner_buffer", "type": "short_term", "config": {"max_items": 5}},
            {"name": "executor_buffer", "type": "short_term", "config": {"max_items": 10}},
        ],
        "connections": [
            {"source": "planner", "target": "executor"},
        ],
        "execution": {"mode": "sequential", "max_iterations": 1, "retry_limit": 1},
    }
    return workflow


async def generate_workflow(prompt: str, adapter: ModelAdapter) -> Dict[str, Any]:
    """Generate a workflow using the meta-agent with fallback validation."""
    try:
        candidate = await _llm_generate(prompt, adapter)
        valid, errors = validate_workflow(candidate)
        if not valid:
            raise ValueError("; ".join(errors))
        return candidate
    except Exception as err:  # pylint: disable=broad-except
        logger.error("Meta-agent generation failed: %s", err, exc_info=True)
        workflow = _fallback_workflow(prompt)
        valid, errors = validate_workflow(workflow)
        if not valid:
            raise RuntimeError(f"Fallback workflow invalid: {errors}")
        return workflow


def generate_workflow_sync(prompt: str, adapter: ModelAdapter) -> Dict[str, Any]:
    """Synchronous helper for CLI usage."""
    return asyncio.run(generate_workflow(prompt, adapter))
