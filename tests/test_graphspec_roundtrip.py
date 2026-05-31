"""Round-trip equivalence: load_graph_spec vs spec_to_python → exec.

The two consumers of a GraphSpec — the direct loader and the Python-source
exporter — must produce equivalent graphs. If a future change to either
breaks structural or behavioural parity, this test catches it.
"""
from __future__ import annotations

import os
from typing import Any

from fleet.agents.agent import Agent
from fleet.core.graph import CompiledGraph
from fleet.core.messages import AgentMessage
from fleet.core.state import GraphState
from fleet.graphspec import GraphSpec, load_graph_spec, register_condition, spec_to_python


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _DeterministicLLM:
    """Returns ``f'{node} done'`` so we can verify which nodes ran."""

    def __init__(self, node_name: str) -> None:
        self._node = node_name

    async def complete(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentMessage:
        return AgentMessage(role="assistant", content=f"{self._node} done")

    @property
    def usage(self) -> None:
        return None


def _swap_agent_llms(cg: CompiledGraph) -> None:
    for name, node in cg.nodes.items():
        fn = node.fn
        owner = getattr(fn, "__self__", None)
        if isinstance(owner, Agent):
            owner.llm = _DeterministicLLM(name)


def _three_node_spec() -> dict:
    return {
        "version": "0.3",
        "name": "rt_team",
        "nodes": [
            {
                "id": "planner",
                "type": "agent",
                "agent": {
                    "name": "planner",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "system": "Break the goal into subqueries.",
                    "tools": [],
                    "memory_bank": False,
                },
                "position": {"x": 100, "y": 40},
            },
            {
                "id": "researcher",
                "type": "agent",
                "agent": {
                    "name": "researcher",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "system": "Research the topic.",
                    "tools": ["web_search"],
                    "memory_bank": False,
                },
                "position": {"x": 300, "y": 40},
            },
            {
                "id": "writer",
                "type": "agent",
                "agent": {
                    "name": "writer",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "system": "Write a summary.",
                    "tools": [],
                    "memory_bank": False,
                },
                "position": {"x": 500, "y": 40},
            },
        ],
        "edges": [
            # Conditional edge: only follow planner→researcher when 'always' fires
            {"src": "planner", "dst": "researcher", "cond": "always"},
            {"src": "researcher", "dst": "writer"},
        ],
        "entry": "planner",
        "exit": "writer",
    }


def _exec_exported(source: str) -> CompiledGraph:
    """Execute spec_to_python output in a fresh namespace and return its graph."""
    ns: dict[str, Any] = {"__name__": "_rt_exported"}
    code = compile(source, "<exported>", "exec")
    exec(code, ns)
    cg = ns["graph"]
    assert isinstance(cg, CompiledGraph), f"exported namespace missing CompiledGraph; got {type(cg)}"
    return cg


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

async def test_roundtrip_structural_equivalence(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build")

    raw = _three_node_spec()
    spec = GraphSpec.model_validate(raw)

    cg_loaded = load_graph_spec(spec)
    cg_exported = _exec_exported(spec_to_python(spec))

    # Same name
    assert cg_loaded.name == cg_exported.name == "rt_team"

    # Same node ids
    assert set(cg_loaded.nodes) == set(cg_exported.nodes) == {"planner", "researcher", "writer"}

    # Same entry / exit
    assert cg_loaded.entry == cg_exported.entry == "planner"
    assert cg_loaded.exit == cg_exported.exit == "writer"

    # Same edges (src, dst, has_cond)
    def _edge_keys(cg: CompiledGraph) -> set[tuple[str, str, bool]]:
        return {(e.src, e.dst, e.cond is not None) for e in cg.edges}

    assert _edge_keys(cg_loaded) == _edge_keys(cg_exported)

    # Same agent configuration on each node (name, tools, max_iters, goal)
    for node_id in cg_loaded.nodes:
        a_loaded = cg_loaded.nodes[node_id].fn.__self__  # type: ignore[attr-defined]
        a_exported = cg_exported.nodes[node_id].fn.__self__  # type: ignore[attr-defined]
        assert isinstance(a_loaded, Agent)
        assert isinstance(a_exported, Agent)
        assert a_loaded.name == a_exported.name
        assert a_loaded.tools == a_exported.tools
        assert a_loaded.max_iters == a_exported.max_iters
        assert a_loaded.goal == a_exported.goal


async def test_roundtrip_behavioural_equivalence(monkeypatch):
    """Same goal + same deterministic LLMs → same final assistant message sequence."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build")

    raw = _three_node_spec()
    spec = GraphSpec.model_validate(raw)

    cg_loaded = load_graph_spec(spec)
    cg_exported = _exec_exported(spec_to_python(spec))

    _swap_agent_llms(cg_loaded)
    _swap_agent_llms(cg_exported)

    final_loaded = await cg_loaded.run(GraphState(goal="hello"))
    final_exported = await cg_exported.run(GraphState(goal="hello"))

    def _assistant_contents(s: GraphState) -> list[str]:
        return [m.content for m in s.messages if m.role == "assistant"]

    assert _assistant_contents(final_loaded) == _assistant_contents(final_exported)
    # Sanity: all three nodes fired in both runs.
    assert _assistant_contents(final_loaded) == ["planner done", "researcher done", "writer done"]


async def test_roundtrip_conditional_edge_blocks_traversal(monkeypatch):
    """A False-valued condition prevents the edge from firing in both paths."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build")

    # Register a never-firing condition just for this test.
    register_condition("rt_never_fires", lambda s: False)

    raw = _three_node_spec()
    # Block planner → researcher; add a fallback planner → writer that always fires.
    raw["edges"] = [
        {"src": "planner", "dst": "researcher", "cond": "rt_never_fires"},
        {"src": "planner", "dst": "writer", "cond": "always"},
        {"src": "researcher", "dst": "writer"},
    ]
    spec = GraphSpec.model_validate(raw)

    cg_loaded = load_graph_spec(spec)
    cg_exported = _exec_exported(spec_to_python(spec))

    _swap_agent_llms(cg_loaded)
    _swap_agent_llms(cg_exported)

    fa = await cg_loaded.run(GraphState(goal="g"))
    fb = await cg_exported.run(GraphState(goal="g"))

    def _seen(s: GraphState) -> set[str]:
        return {m.content for m in s.messages if m.role == "assistant"}

    assert _seen(fa) == _seen(fb)
    assert "researcher done" not in _seen(fa)
    assert "writer done" in _seen(fa)


# Belt-and-braces: ensure the conditional-edge spec also passes a behavioural
# roundtrip with the 'always' condition active.
async def test_roundtrip_with_always_condition(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build")
    # Touch os to silence pyflakes on the unused import in tighter linters.
    assert os.environ.get("ANTHROPIC_API_KEY")

    raw = _three_node_spec()
    spec = GraphSpec.model_validate(raw)

    cg_loaded = load_graph_spec(spec)
    cg_exported = _exec_exported(spec_to_python(spec))

    _swap_agent_llms(cg_loaded)
    _swap_agent_llms(cg_exported)

    fa = await cg_loaded.run(GraphState(goal="g"))
    fb = await cg_exported.run(GraphState(goal="g"))

    assert [m.content for m in fa.messages if m.role == "assistant"] == [
        m.content for m in fb.messages if m.role == "assistant"
    ]
