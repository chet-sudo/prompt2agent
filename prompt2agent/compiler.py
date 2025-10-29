"""Compilation utilities that transform prompts into workflow specifications."""

from __future__ import annotations

import json
import logging
import uuid
from agents import Agent, Runner
from agents.agent_output import AgentOutputSchema

from config import ProviderConfig, ensure_provider_config
from models import AgentPlan, AgentSpec, ExecutionSpec, ToolSpec, WorkflowMetadata, WorkflowPlan, WorkflowSpec

logger = logging.getLogger(__name__)

_DESIGNER_PROMPT = """
You are WorkflowSmith, an architect that builds the minimal set of collaborating agents
required to answer a user prompt. Design agents that follow the OpenAI Agents SDK conventions:
- Prefer hosted tools when available. "web_search" is the first choice for factual research.
- Only generate a "python_repl" tool when lightweight computation is absolutely required.
- Use "agent_tool" when one agent should invoke another agent as a tool. Do not create
  redundant chains—keep the graph simple and focused on the goal.
- Each agent needs clear, actionable instructions and a concise summary of its role.
- Exactly one agent must have "final": true. This agent issues the final user-facing answer.
- Set "entrypoint" to the agent that should start the run (often the coordinator/final agent).
- Keep the agent list as small as possible while still covering the required capabilities.
Return JSON that satisfies the WorkflowPlan schema. Only emit the JSON payload.
"""


def _plan_to_spec(plan: WorkflowPlan, prompt: str, provider: ProviderConfig) -> WorkflowSpec:
    agents = [_plan_agent_to_spec(agent) for agent in plan.agents]

    if not any(agent.final for agent in agents):
        logger.debug("No agent marked final in plan; promoting entrypoint as final responder.")
        for agent in agents:
            if agent.key == plan.entrypoint:
                agent.final = True
                break

    metadata = WorkflowMetadata(
        spec_id=uuid.uuid4().hex,
        title=plan.title,
        summary=plan.summary,
        source_prompt=prompt,
        default_model=provider.model_id,
    )

    execution = ExecutionSpec(entrypoint=plan.entrypoint, max_turns=plan.max_turns)

    return WorkflowSpec(metadata=metadata, agents=agents, execution=execution)


def _plan_agent_to_spec(plan: AgentPlan) -> AgentSpec:
    return AgentSpec(
        key=plan.key,
        name=plan.name,
        summary=plan.summary,
        instructions=plan.instructions,
        handoff_description=plan.summary,
        tools=[_ensure_tool_defaults(tool) for tool in plan.tools],
        handoffs=list(plan.handoffs),
        final=plan.final,
    )


def _ensure_tool_defaults(tool: ToolSpec) -> ToolSpec:
    if tool.kind == "web_search" and not tool.name:
        tool.name = "web_search"
    return tool


def transform_prompt_to_workflow(prompt: str) -> WorkflowSpec:
    """Compile a natural language prompt into a serialisable workflow specification."""

    provider_config = ensure_provider_config()

    schema = WorkflowPlan.model_json_schema()
    instructions = _DESIGNER_PROMPT + "\nSchema:\n" + json.dumps(schema, indent=2)

    planner_agent = Agent(
        name="workflow_designer",
        handoff_description="Designs agent workflows",
        instructions=instructions,
        output_type=AgentOutputSchema(WorkflowPlan),
        model=provider_config.model_id,
    )

    run_config = provider_config.build(workflow_name="Workflow compilation")

    logger.info("Synthesising workflow plan for prompt: %s", prompt)
    result = Runner.run_sync(planner_agent, prompt, run_config=run_config)
    plan = result.final_output

    if not isinstance(plan, WorkflowPlan):
        raise TypeError("Planner output is not a WorkflowPlan instance.")

    logger.debug("Workflow plan produced: %s", plan.model_dump())
    spec = _plan_to_spec(plan, prompt, provider_config)
    logger.info("Compiled workflow '%s' with %d agents.", spec.metadata.title, len(spec.agents))
    return spec


__all__ = ["transform_prompt_to_workflow"]
