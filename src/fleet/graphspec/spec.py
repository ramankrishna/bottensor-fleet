"""Pydantic models for the Fleet JSON graph format (v0.3)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SUPPORTED_PROVIDERS: frozenset[str] = frozenset(
    {"anthropic", "claude", "openai", "deepseek", "custom", "ollama", "gemini", "mistral"}
)
"""LLM providers the loader knows how to construct FleetLLMs for.

``custom`` covers any OpenAI-compatible endpoint (vLLM, Together, Ollama-as-OpenAI,
LM Studio, etc.) and requires ``base_url`` to be set. ``deepseek`` is a hosted
OpenAI-compatible provider; ``base_url`` is optional and defaults to the
DeepSeek API endpoint.
"""


class Position(BaseModel):
    """Optional UI coordinates for graph editors."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class AgentSpec(BaseModel):
    """Configuration for a single agent node."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    model: str
    system: str = ""
    tools: list[str] = Field(default_factory=list)
    memory_bank: bool = False
    base_url: str | None = None
    max_iters: int = 10

    @model_validator(mode="after")
    def _check_provider(self) -> "AgentSpec":
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}."
            )
        if self.provider == "custom" and not self.base_url:
            raise ValueError(
                "provider='custom' requires base_url to point at an "
                "OpenAI-compatible endpoint (e.g. http://localhost:11434/v1)."
            )
        return self

    @model_validator(mode="after")
    def _check_tools(self) -> "AgentSpec":
        # Import here so that the tool registry is fully populated as a side
        # effect of `import fleet.tools` (which registers all built-ins).
        from fleet.tools.base import get_tool

        unknown = [t for t in self.tools if get_tool(t) is None]
        if unknown:
            raise ValueError(
                f"Unknown tool name(s): {unknown}. "
                "Register the tool with @fleet.tool before referencing it in a graph spec."
            )
        return self


class NodeSpec(BaseModel):
    """A node in the graph. For v0.3, only ``type='agent'`` is supported."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["agent"] = "agent"
    agent: AgentSpec
    position: Position | None = None


class EdgeSpec(BaseModel):
    """An edge from ``src`` → ``dst``, optionally gated by a named condition."""

    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str
    cond: str | None = None


class GraphSpec(BaseModel):
    """Top-level graph specification."""

    model_config = ConfigDict(extra="forbid")

    version: str
    name: str
    nodes: list[NodeSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    entry: str
    exit: str

    @model_validator(mode="after")
    def _check_topology(self) -> "GraphSpec":
        node_ids = {n.id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            seen: set[str] = set()
            dupes: list[str] = []
            for n in self.nodes:
                if n.id in seen:
                    dupes.append(n.id)
                seen.add(n.id)
            raise ValueError(f"Duplicate node id(s): {sorted(set(dupes))}.")

        if self.entry not in node_ids:
            raise ValueError(
                f"entry='{self.entry}' is not in nodes ({sorted(node_ids)})."
            )
        if self.exit not in node_ids:
            raise ValueError(
                f"exit='{self.exit}' is not in nodes ({sorted(node_ids)})."
            )

        for edge in self.edges:
            if edge.src not in node_ids:
                raise ValueError(
                    f"edge src='{edge.src}' references unknown node "
                    f"(known: {sorted(node_ids)})."
                )
            if edge.dst not in node_ids:
                raise ValueError(
                    f"edge dst='{edge.dst}' references unknown node "
                    f"(known: {sorted(node_ids)})."
                )

        # Condition names must be registered (no eval, no arbitrary code).
        from fleet.graphspec.conditions import get_condition

        for edge in self.edges:
            if edge.cond is None:
                continue
            if get_condition(edge.cond) is None:
                raise ValueError(
                    f"edge {edge.src}->{edge.dst} references unknown "
                    f"condition '{edge.cond}'. Register it via "
                    "fleet.graphspec.register_condition()."
                )

        return self
