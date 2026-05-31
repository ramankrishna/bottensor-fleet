"""Pydantic models for the Fleet JSON graph format (v0.3)."""
from __future__ import annotations

import urllib.parse
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# SSRF guard for base_url
# ---------------------------------------------------------------------------
#
# A spec authored in the visual builder is untrusted: it flows in over the
# network and is passed to load_graph_spec(). ``base_url`` is forwarded to
# the OpenAI SDK as the API endpoint, so a malicious value can be used to
# hit cloud metadata services (169.254.169.254, metadata.google.internal,
# …) and exfiltrate IAM credentials, or to coax the runtime into reading
# local files (file://, gopher://).
#
# We block:
#   - any scheme other than http / https
#   - known cloud metadata endpoints (AWS, GCP, Alibaba, Azure IMDS)
#
# We deliberately ALLOW localhost / 127.0.0.1 / 0.0.0.0 / ::1 — this is the
# whole point of provider='custom'. Local vLLM, Ollama, LM Studio etc.
# live there, and blocking them would defeat the feature. Operators who
# need stricter network isolation should run fleet behind a sandbox or
# egress firewall; this validator catches the obvious shapes only.

_BLOCKED_METADATA_HOSTS: frozenset[str] = frozenset({
    # AWS / OpenStack instance metadata
    "169.254.169.254",
    # AWS ECS task metadata
    "169.254.170.2",
    # Google Cloud instance metadata
    "metadata.google.internal",
    "metadata.goog",
    # Alibaba Cloud
    "100.100.100.200",
    # Azure Instance Metadata Service
    "169.254.169.253",
})


def validate_base_url(url: str) -> None:
    """Reject base_url values that target cloud-metadata SSRF endpoints or
    non-http(s) schemes. Localhost is explicitly allowed (local LLMs).

    Raises ``ValueError`` with a clear message on rejection. Pure function —
    no network calls.
    """
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(
            f"base_url must use http or https; got scheme '{scheme}'. "
            "(Non-http schemes like file:// and gopher:// are blocked to "
            "prevent SSRF.)"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("base_url must include a hostname.")
    if host in _BLOCKED_METADATA_HOSTS:
        raise ValueError(
            f"base_url host '{host}' is a cloud-metadata endpoint and is "
            "blocked. Visit AWS/GCP/Azure instance metadata services from "
            "your own code, not from a graph spec."
        )

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
        if self.base_url:
            validate_base_url(self.base_url)
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
