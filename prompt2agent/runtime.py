"""Runtime helpers for executing compiled workflows."""

from __future__ import annotations

import logging

from agents import Runner
from agents.memory import SQLiteSession

from persistence import session_store_path
from workflow import RunnableWorkflow

logger = logging.getLogger(__name__)


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
