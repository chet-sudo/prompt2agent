"""Pydantic models that capture workflow specifications and compilation plans."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldValidationInfo,
    field_validator,
    model_validator,
)


class ToolSpec(BaseModel):
    """Describes a tool that an agent can use."""

    kind: Literal["web_search", "agent_tool", "python_repl"]
    description: str = Field(..., description="Purpose of the tool and when to use it.")
    name: str = Field(..., description="Short human readable name for the tool.")
    target_agent: Optional[str] = Field(
        default=None,
        description="Agent key that should be invoked when this is an agent-as-tool reference.",
    )
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional tool settings.")

    @field_validator("target_agent")
    @classmethod
    def _validate_target_for_agent_tool(
        cls, value: Optional[str], info: FieldValidationInfo
    ) -> Optional[str]:
        kind = info.data.get("kind") if isinstance(info.data, dict) else None
        if kind == "agent_tool" and not value:
            raise ValueError("agent_tool entries must declare a target_agent")
        return value


class HandoffSpec(BaseModel):
    """Represents an optional handoff between agents."""

    target_agent: str
    description: str


class AgentPlan(BaseModel):
    """Description of an agent as produced by the planning LLM."""

    key: str = Field(..., description="Machine-friendly unique identifier for the agent.")
    name: str = Field(..., description="Human readable name for the agent.")
    summary: str = Field(..., description="Short summary of the agent's responsibility.")
    instructions: str = Field(
        ..., description="Detailed instructions/system prompt for the agent to follow."
    )
    capabilities: List[str] = Field(default_factory=list, description="Semantic capability tags.")
    tools: List[ToolSpec] = Field(default_factory=list)
    handoffs: List[HandoffSpec] = Field(default_factory=list)
    final: bool = Field(
        default=False,
        description="Whether this agent must deliver the final user facing answer.",
    )

    model_config = ConfigDict(populate_by_name=True)


class WorkflowPlan(BaseModel):
    """High level workflow description returned by the planning agent."""

    title: str
    summary: str
    entrypoint: str = Field(..., description="The key of the agent that should start the run.")
    max_turns: int = Field(
        default=18,
        ge=4,
        le=40,
        description="Maximum dialogue turns the runner should allow before aborting.",
    )
    agents: List[AgentPlan]

    @field_validator("agents")
    @classmethod
    def _ensure_unique_keys(cls, value: List[AgentPlan]) -> List[AgentPlan]:
        keys = {agent.key for agent in value}
        if len(keys) != len(value):
            raise ValueError("Agent keys must be unique within a plan.")
        return value

    @model_validator(mode="after")
    def _validate_entrypoint(self) -> "WorkflowPlan":
        if self.entrypoint not in {agent.key for agent in self.agents}:
            raise ValueError("Entrypoint must reference an agent key defined in the plan.")
        return self


class WorkflowMetadata(BaseModel):
    """Metadata stored with every workflow specification."""

    spec_id: str
    title: str
    summary: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_prompt: str
    default_model: str

    model_config = ConfigDict(json_encoders={datetime: lambda value: value.isoformat()})


class AgentSpec(BaseModel):
    """Materialised agent definition stored in the workflow specification."""

    key: str
    name: str
    summary: str
    instructions: str
    handoff_description: Optional[str] = None
    model: Optional[str] = Field(
        default=None, description="Optional override model identifier for this agent."
    )
    tools: List[ToolSpec] = Field(default_factory=list)
    handoffs: List[HandoffSpec] = Field(default_factory=list)
    final: bool = False


class ExecutionSpec(BaseModel):
    """Execution parameters for the workflow."""

    entrypoint: str
    max_turns: int = Field(default=18, ge=4, le=40)


class WorkflowSpec(BaseModel):
    """Serializable workflow specification persisted on disk."""

    metadata: WorkflowMetadata
    agents: List[AgentSpec]
    execution: ExecutionSpec

    model_config = ConfigDict(json_encoders={datetime: lambda value: value.isoformat()})


__all__ = [
    "AgentPlan",
    "AgentSpec",
    "ExecutionSpec",
    "HandoffSpec",
    "ToolSpec",
    "WorkflowMetadata",
    "WorkflowPlan",
    "WorkflowSpec",
]
