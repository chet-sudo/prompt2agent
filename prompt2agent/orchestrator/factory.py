"""Factory utilities to instantiate workflow agents."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from openai_agents import Agent as OpenAIAgent  # type: ignore
except Exception:  # pragma: no cover - fallback
    OpenAIAgent = object  # type: ignore[misc,assignment]

from prompt2agent.adapters.model_adapter import ModelAdapter
from prompt2agent.memory.short_term import ShortTermMemoryBuffer
from prompt2agent.tools.base import ToolDefinition, ToolRegistry
from prompt2agent.tools.web_search import web_search as web_search_definition
from prompt2agent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentWrapper:
    """Wrapper bridging workflow agents with the model adapter."""

    id: str
    name: str
    instructions: str
    model_adapter: ModelAdapter
    tools: Dict[str, ToolDefinition]
    memory: Dict[str, ShortTermMemoryBuffer]
    model_override: Optional[str] = None

    async def run(self, *, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with the provided input text."""
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": input_text},
        ]
        memory_dump: List[str] = []
        for buffer in self.memory.values():
            memory_dump.extend(buffer.dump())
        if memory_dump:
            messages.insert(1, {"role": "system", "content": "Memory: " + "\n".join(memory_dump)})
        logger.debug("Agent %s dispatching messages: %s", self.id, messages)
        extra_body: Dict[str, Any] | None = None
        if self.model_override:
            extra_body = {"model": self.model_override}
        response = await self.model_adapter.chat(messages, extra_body=extra_body)
        logger.debug("Agent %s response: %s", self.id, response)
        for buffer in self.memory.values():
            buffer.add(response)
        return {"agent_id": self.id, "output": response, "context": context}

    def run_sync(self, *, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.run(input_text=input_text, context=context))


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(web_search_definition)
    return registry


def instantiate_tools(workflow_tools: Iterable[Dict[str, Any]], registry: ToolRegistry) -> Dict[str, ToolDefinition]:
    tools: Dict[str, ToolDefinition] = {}
    for entry in workflow_tools:
        name = entry["name"]
        tools[name] = registry.get(name)
    return tools


def instantiate_memory(definitions: Iterable[Dict[str, Any]]) -> Dict[str, ShortTermMemoryBuffer]:
    memories: Dict[str, ShortTermMemoryBuffer] = {}
    for entry in definitions:
        if entry.get("type") != "short_term":
            raise ValueError(f"Unsupported memory type: {entry.get('type')}")
        config = entry.get("config", {})
        max_items = int(config.get("max_items", 10))
        memories[entry["name"]] = ShortTermMemoryBuffer(max_items=max_items)
    return memories


def create_agents(
    workflow: Dict[str, Any],
    *,
    model_adapter: ModelAdapter,
) -> Dict[str, AgentWrapper]:
    """Instantiate agents from workflow definition."""
    tool_registry = build_tool_registry()
    tool_configs = {tool["name"]: tool for tool in workflow.get("tools", [])}
    memory_configs = {mem["name"]: mem for mem in workflow.get("memory", [])}

    agents: Dict[str, AgentWrapper] = {}
    for agent_def in workflow["agents"]:
        agent_tools = {
            name: tool_registry.get(name)
            for name in agent_def.get("tools", [])
            if name in tool_configs
        }
        agent_memory = {
            name: instantiate_memory([memory_configs[name]])[name]
            for name in agent_def.get("memory", [])
            if name in memory_configs
        }
        wrapper = AgentWrapper(
            id=agent_def["id"],
            name=agent_def["name"],
            instructions=agent_def["instructions"],
            model_adapter=model_adapter,
            tools=agent_tools,
            memory=agent_memory,
            model_override=agent_def.get("model"),
        )
        agents[wrapper.id] = wrapper
        logger.debug("Instantiated agent %s", wrapper)
    return agents
