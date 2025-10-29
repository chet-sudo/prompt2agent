"""Utilities to materialise workflow specifications into runnable agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from agents import Agent, Tool

from config import ProviderConfig, ensure_provider_config
from models import ToolSpec, WorkflowSpec
from tools import attach_agent_tools, build_base_tools

logger = logging.getLogger(__name__)


@dataclass
class RunnableWorkflow:
    """Wraps a materialised workflow and the associated runtime configuration."""

    spec: WorkflowSpec
    provider: ProviderConfig
    agents: Dict[str, Agent]

    @property
    def entrypoint(self) -> Agent:
        return self.agents[self.spec.execution.entrypoint]

    @property
    def final_agents(self) -> List[str]:
        return [agent.key for agent in self.spec.agents if agent.final]


def build_agents(spec: WorkflowSpec, provider: ProviderConfig) -> Dict[str, Agent]:
    agents: Dict[str, Agent] = {}
    deferred_tools: List[tuple[str, ToolSpec, str]] = []

    for agent_spec in spec.agents:
        base_tools: List[Tool] = []
        for tool_spec in agent_spec.tools:
            result = build_base_tools(tool_spec)
            base_tools.extend(result.tools)
            for deferred_spec, target_key in result.deferred_agent_tools:
                deferred_tools.append((agent_spec.key, deferred_spec, target_key))

        agent = Agent(
            name=agent_spec.name,
            handoff_description=agent_spec.summary,
            instructions=agent_spec.instructions,
            tools=list(base_tools),
            model=agent_spec.model or provider.model_id,
        )
        agents[agent_spec.key] = agent

    for owner_key, tool_spec, target_key in deferred_tools:
        if target_key not in agents:
            raise KeyError(f"Agent tool target '{target_key}' is missing from the specification.")
        owner = agents[owner_key]
        target = agents[target_key]
        attach_agent_tools(owner, tool_spec, target)
        logger.debug("Bound agent '%s' as tool '%s' for '%s'.", target_key, tool_spec.name, owner_key)

    return agents


def materialise_workflow(spec: WorkflowSpec) -> RunnableWorkflow:
    provider = ensure_provider_config()
    agents = build_agents(spec, provider)
    return RunnableWorkflow(spec=spec, provider=provider, agents=agents)


__all__ = ["RunnableWorkflow", "build_agents", "materialise_workflow"]
