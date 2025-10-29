"""Runtime helpers for executing compiled workflows."""

from __future__ import annotations

import logging
from typing import Iterable

from agents import Agent, Runner
from agents.memory import SQLiteSession

try:  # pragma: no cover - optional dependency errors are handled at runtime
    from litellm import NotFoundError
except ImportError:  # pragma: no cover - defensive fallback if litellm is unavailable
    NotFoundError = RuntimeError  # type: ignore[assignment]

from .persistence import session_store_path
from .workflow import RunnableWorkflow

logger = logging.getLogger(__name__)


def _gather_collaborator_instructions(workflow: RunnableWorkflow, entry_key: str) -> Iterable[str]:
    """Yield descriptive instructions for collaborators of the entry agent."""

    for agent in workflow.spec.agents:
        if agent.key == entry_key:
            continue
        collaborator = [f"- {agent.name}: {agent.summary or agent.instructions}".strip()]
        extra = agent.instructions.strip()
        if extra:
            collaborator.append(f"  Guidance: {extra}")
        yield "\n".join(collaborator)


def _run_single_agent_fallback(workflow: RunnableWorkflow) -> str:
    """Execute a simplified single-agent version of the workflow when tools are unavailable."""

    entry_key = workflow.spec.execution.entrypoint
    entry_spec = next(agent for agent in workflow.spec.agents if agent.key == entry_key)

    instructions_parts = [entry_spec.instructions.strip()]
    instructions_parts.append(
        "You cannot delegate work to other agents or call external tools. Satisfy the request independently, "
        "drawing on your own reasoning."
    )

    collaborator_snippets = list(_gather_collaborator_instructions(workflow, entry_key))
    if collaborator_snippets:
        instructions_parts.append(
            "Incorporate the responsibilities originally assigned to your collaborators:\n" + "\n".join(collaborator_snippets)
        )

    fallback_agent = Agent(
        name=f"{entry_spec.name} (standalone)",
        handoff_description=entry_spec.summary,
        instructions="\n\n".join(part for part in instructions_parts if part),
        tools=[],
        model=entry_spec.model or workflow.provider.model_id,
    )

    run_config = workflow.provider.build(workflow_name=f"{workflow.spec.metadata.title} (fallback)")
    result = Runner.run_sync(
        fallback_agent,
        workflow.spec.metadata.source_prompt,
        run_config=run_config,
    )

    output = result.final_output
    if not isinstance(output, str):
        output = str(output)
    if not output.strip():
        raise ValueError("Fallback execution completed without a final answer.")

    logger.info("Fallback workflow run completed with single agent '%s'.", fallback_agent.name)
    return output


def _is_tool_capability_error(error: Exception) -> bool:
    message = str(error).lower()
    return "support tool use" in message or "tool use" in message


def run_workflow(workflow: RunnableWorkflow) -> str:
    """Execute a materialised workflow and return the final answer string."""

    workflow_id = workflow.spec.metadata.spec_id
    session_path = session_store_path(workflow_id)
    session = SQLiteSession(session_id=workflow_id, db_path=session_path)
    run_config = workflow.provider.build(workflow_name=workflow.spec.metadata.title)

    logger.info(
        "Running workflow '%s' with entrypoint '%s'", workflow.spec.metadata.title, workflow.spec.execution.entrypoint
    )

    try:
        result = Runner.run_sync(
            workflow.entrypoint,
            workflow.spec.metadata.source_prompt,
            run_config=run_config,
            session=session,
        )
    except NotFoundError as exc:  # pragma: no cover - depends on provider error handling
        if _is_tool_capability_error(exc):
            logger.warning("Model does not support tool use; falling back to single-agent execution.")
            return _run_single_agent_fallback(workflow)
        raise
    finally:
        session.close()

    output = result.final_output
    if not isinstance(output, str):
        output = str(output)
    if not output.strip():
        raise ValueError("Workflow execution completed without a final answer.")

    logger.info("Workflow run completed with final agent '%s'.", result.last_agent.name)
    return output


__all__ = ["run_workflow"]
