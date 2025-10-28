"""Tool base abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Protocol


class ToolCallable(Protocol):
    """Protocol for async tool callables."""

    async def __call__(self, *, query: str) -> Dict[str, Any]:  # pragma: no cover - protocol
        ...


@dataclass(slots=True)
class ToolDefinition:
    """Definition used when wiring tools to agents."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    callable: ToolCallable


class ToolRegistry:
    """Registry for pluggable tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def all(self) -> Dict[str, ToolDefinition]:
        return dict(self._tools)


def tool_definition(
    *,
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
) -> Callable[[ToolCallable], ToolDefinition]:
    """Decorator to create a tool definition."""

    def decorator(func: ToolCallable) -> ToolDefinition:
        return ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            callable=func,
        )

    return decorator
