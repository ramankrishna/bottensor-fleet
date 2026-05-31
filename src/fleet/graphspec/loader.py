"""Load a JSON ``GraphSpec`` into a runnable :class:`fleet.CompiledGraph`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from fleet.agents.agent import Agent
from fleet.core.graph import CompiledGraph, Graph
from fleet.graphspec.conditions import get_condition
from fleet.graphspec.spec import AgentSpec, GraphSpec
from fleet.providers.client import FleetLLM

SpecLike = Union[GraphSpec, dict, str, Path]


def load_graph_spec(spec: SpecLike, *, backend: Any = None) -> CompiledGraph:
    """Turn a JSON graph spec into a runnable :class:`CompiledGraph`.

    ``spec`` may be:
        - a :class:`GraphSpec` instance,
        - a ``dict`` matching the schema,
        - a JSON string,
        - or a path (``str``/``Path``) to a ``.json`` file.

    The optional ``backend`` argument is forwarded to ``Graph.compile`` —
    pass ``"sqlite"``, ``"redis"``, or a custom CheckpointBackend instance.
    """
    gs = _coerce_to_spec(spec)
    graph = _build_graph(gs)
    return graph.compile(backend=backend)


def _coerce_to_spec(spec: SpecLike) -> GraphSpec:
    if isinstance(spec, GraphSpec):
        return spec
    if isinstance(spec, dict):
        return GraphSpec.model_validate(spec)
    if isinstance(spec, (str, Path)):
        text: str
        path_str = str(spec)
        # Treat as JSON literal if it looks like JSON; otherwise as a path.
        looks_like_json = path_str.lstrip().startswith("{")
        if looks_like_json:
            text = path_str
        else:
            text = Path(path_str).read_text(encoding="utf-8")
        return GraphSpec.model_validate(json.loads(text))
    raise TypeError(
        f"Unsupported spec type: {type(spec).__name__}. "
        "Pass a GraphSpec, dict, JSON string, or path to a .json file."
    )


def _build_agent(agent_spec: AgentSpec) -> Agent:
    """Construct a FleetLLM and Agent from an AgentSpec."""
    llm_kwargs: dict[str, Any] = {}
    if agent_spec.base_url:
        llm_kwargs["base_url"] = agent_spec.base_url

    llm = FleetLLM(agent_spec.provider, agent_spec.model, **llm_kwargs)

    memory_bank = None
    if agent_spec.memory_bank:
        from fleet.memory.bank import ReasoningBank

        memory_bank = ReasoningBank()

    return Agent(
        llm=llm,
        tools=list(agent_spec.tools),
        max_iters=agent_spec.max_iters,
        name=agent_spec.name,
        goal=agent_spec.system,
        memory_bank=memory_bank,
    )


def _build_graph(gs: GraphSpec) -> Graph:
    graph = Graph(gs.name)

    for node in gs.nodes:
        agent = _build_agent(node.agent)
        graph.add_node(node.id, agent.step)

    for edge in gs.edges:
        cond_fn = get_condition(edge.cond) if edge.cond else None
        graph.add_edge(edge.src, edge.dst, cond=cond_fn)

    graph.set_entry(gs.entry).set_exit(gs.exit)
    return graph
