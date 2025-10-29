"""Hosted and custom tool bindings used by compiled workflows."""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict

from agents import Agent, Tool, WebSearchTool, function_tool

from .models import ToolSpec

_SAFE_GLOBALS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "sorted": sorted,
        }
    }
)


@function_tool(name_override="python_repl", description_override="Execute short Python snippets.")
def python_repl(code: str) -> str:
    """Execute a short Python snippet and return the captured output.

    The code executes with a restricted set of built-ins. Assign the final result to a
    variable called ``result`` to make the response explicit.
    """

    if not isinstance(code, str):
        raise ValueError("Code must be a string snippet.")

    cleaned = textwrap.dedent(code).strip()
    if not cleaned:
        return ""

    try:
        ast.parse(cleaned)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python snippet: {exc}") from exc

    local_ns: Dict[str, Any] = {}
    exec(cleaned, dict(_SAFE_GLOBALS), local_ns)
    if "result" in local_ns:
        return str(local_ns["result"])
    return "Snippet executed. Define `result` to provide an explicit answer."


@dataclass
class ToolBuildResult:
    tools: list[Tool]
    deferred_agent_tools: list[tuple[ToolSpec, str]]


def build_base_tools(spec: ToolSpec) -> ToolBuildResult:
    """Instantiate hosted/custom tools that do not depend on other agents."""

    if spec.kind == "web_search":
        return ToolBuildResult([WebSearchTool()], [])

    if spec.kind == "python_repl":
        return ToolBuildResult([python_repl], [])

    if spec.kind == "agent_tool":
        if not spec.target_agent:
            raise ValueError('agent_tool requires a target_agent')
        return ToolBuildResult([], [(spec, spec.target_agent)])

    raise ValueError(f"Unsupported tool kind: {spec.kind}")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return cleaned.lower() or "agent_tool"


def attach_agent_tools(
    owner: Agent,
    spec: ToolSpec,
    target: Agent,
) -> Tool:
    """Create an agent-as-tool binding and attach it to the owner."""

    tool_name = spec.name or _slugify(f"{target.name}_tool")
    tool_description = spec.description or f"Delegate to {target.name}."
    new_tool = target.as_tool(tool_name, tool_description)
    owner.tools.append(new_tool)
    return new_tool


__all__ = [
    "ToolBuildResult",
    "attach_agent_tools",
    "build_base_tools",
    "python_repl",
]
