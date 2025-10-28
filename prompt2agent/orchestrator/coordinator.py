"""Workflow orchestration and execution."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from prompt2agent.adapters.model_adapter import ModelAdapter
from prompt2agent.config import RUNS_DIR, ensure_storage
from prompt2agent.orchestrator.factory import AgentWrapper, create_agents
from prompt2agent.utils.logging import get_logger
from prompt2agent.workflow.schema import validate_workflow

logger = get_logger(__name__)


@dataclass
class RunArtifacts:
    """Paths of saved run artifacts."""

    log_file: Path
    result_file: Path


async def _execute_agent(
    agent: AgentWrapper,
    *,
    input_text: str,
    context: Dict[str, Any],
    retry_limit: int,
) -> Dict[str, Any]:
    last_exception: Exception | None = None
    for attempt in range(retry_limit + 1):
        try:
            logger.debug("Running agent %s attempt %s", agent.id, attempt + 1)
            return await agent.run(input_text=input_text, context=context)
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Agent %s failed: %s", agent.id, err, exc_info=True)
            last_exception = err
    raise RuntimeError(f"Agent {agent.id} failed after {retry_limit + 1} attempts") from last_exception


async def _sequential_execution(
    agents: List[AgentWrapper],
    *,
    goal: str,
    execution: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    logs: List[Dict[str, Any]] = []
    current_input = goal
    retry_limit = int(execution.get("retry_limit", 0))
    for agent in agents:
        entry = await _execute_agent(
            agent,
            input_text=current_input,
            context={"mode": "sequential"},
            retry_limit=retry_limit,
        )
        logs.append(entry)
        current_input = str(entry["output"])
    return logs, current_input


async def _peer_execution(
    agents: List[AgentWrapper],
    *,
    goal: str,
    execution: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    logs: List[Dict[str, Any]] = []
    shared_state = goal
    retry_limit = int(execution.get("retry_limit", 0))
    iterations = int(execution.get("max_iterations", 1))
    for iteration in range(iterations):
        logger.debug("Peer iteration %s", iteration + 1)
        results = await asyncio.gather(
            *[
                _execute_agent(
                    agent,
                    input_text=shared_state,
                    context={"mode": "peer", "iteration": iteration + 1},
                    retry_limit=retry_limit,
                )
                for agent in agents
            ]
        )
        logs.extend(results)
        shared_state = "\n".join(str(result["output"]) for result in results)
    return logs, shared_state


async def run_workflow(
    workflow: Dict[str, Any],
    *,
    model_adapter: ModelAdapter,
) -> Tuple[List[Dict[str, Any]], str, RunArtifacts]:
    """Validate, instantiate and run the workflow."""
    ensure_storage()
    valid, errors = validate_workflow(workflow)
    if not valid:
        raise ValueError(f"Workflow validation failed: {errors}")

    agents_map = create_agents(workflow, model_adapter=model_adapter)
    ordered_agents = [agents_map[agent["id"]] for agent in workflow["agents"]]

    execution = workflow.get("execution", {"mode": "sequential"})
    mode = execution.get("mode", "sequential")
    goal = workflow.get("metadata", {}).get("goal", "")

    if mode == "peer":
        logs, result = await _peer_execution(ordered_agents, goal=goal, execution=execution)
    else:
        logs, result = await _sequential_execution(ordered_agents, goal=goal, execution=execution)

    artifacts = _save_run_artifacts(workflow, logs, result)
    return logs, result, artifacts


def _save_run_artifacts(
    workflow: Dict[str, Any],
    logs: List[Dict[str, Any]],
    result: str,
) -> RunArtifacts:
    ensure_storage()
    metadata = workflow.get("metadata", {})
    workflow_id = metadata.get("id", uuid4().hex)
    goal = metadata.get("goal", "workflow")
    slug_goal = goal.replace(" ", "-").lower()
    run_id = uuid4().hex[:8]
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    run_dir = RUNS_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"run_{slug_goal}_{workflow_id}_{run_id}_{timestamp}"
    log_path = run_dir / f"{base_name}_log.json"
    result_path = run_dir / f"{base_name}_result.json"

    log_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps({"result": result}, indent=2), encoding="utf-8")
    logger.info("Run artifacts saved: %s, %s", log_path, result_path)
    return RunArtifacts(log_file=log_path, result_file=result_path)


def run_workflow_sync(
    workflow: Dict[str, Any],
    *,
    model_adapter: ModelAdapter,
) -> Tuple[List[Dict[str, Any]], str, RunArtifacts]:
    """Synchronous helper for orchestrator."""
    return asyncio.run(run_workflow(workflow, model_adapter=model_adapter))
