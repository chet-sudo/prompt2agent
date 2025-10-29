"""Hosted and custom tool bindings used by compiled workflows."""

from __future__ import annotations

import ast
import logging
import re
import textwrap
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Dict

from agents import Agent, Tool, WebSearchTool, function_tool
from agents.tool import FunctionTool

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

logger = logging.getLogger(__name__)


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


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _configure_function_tool(tool: Tool, config: Dict[str, Any]) -> Tool:
    """Return a tool instance with configuration overrides applied."""

    if not isinstance(tool, FunctionTool):
        return tool

    overrides: Dict[str, Any] = {}

    if "strict_json_schema" in config:
        overrides["strict_json_schema"] = _coerce_bool(config["strict_json_schema"])
    elif "strict_mode" in config:
        overrides["strict_json_schema"] = _coerce_bool(config["strict_mode"])

    if not overrides:
        return tool

    new_tool = replace(tool)
    for key, value in overrides.items():
        setattr(new_tool, key, value)
    return new_tool


def _build_web_search_tool(config: Dict[str, Any]) -> WebSearchTool:
    allowed_keys = {"user_location", "filters", "search_context_size"}
    kwargs = {key: config[key] for key in allowed_keys if key in config}
    return WebSearchTool(**kwargs)


def build_base_tools(spec: ToolSpec) -> ToolBuildResult:
    """Instantiate hosted/custom tools that do not depend on other agents."""

    if spec.kind == "web_search":
        tool = _build_web_search_tool(spec.config)
        return ToolBuildResult([tool], [])

    if spec.kind == "python_repl":
        tool = _configure_function_tool(python_repl, spec.config)
        return ToolBuildResult([tool], [])

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
    config = dict(spec.config)

    agent_tool_kwargs: Dict[str, Any] = {}
    if "max_turns" in config:
        try:
            agent_tool_kwargs["max_turns"] = int(config["max_turns"])
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.warning(
                "Invalid max_turns '%s' for agent tool '%s'; using default.",
                config["max_turns"],
                tool_name,
            )

    new_tool = target.as_tool(tool_name, tool_description, **agent_tool_kwargs)
    configured_tool = _configure_function_tool(new_tool, config)
    owner.tools.append(configured_tool)
    return configured_tool


__all__ = [
    "ToolBuildResult",
    "attach_agent_tools",
    "build_base_tools",
    "python_repl",
]
