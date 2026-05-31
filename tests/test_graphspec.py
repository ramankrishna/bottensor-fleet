"""Tests for the JSON graph spec: schema, validation, loader, CLI integration."""
from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from fleet.cli import app
from fleet.core.messages import AgentMessage
from fleet.core.state import GraphState
from fleet.graphspec import (
    GraphSpec,
    Position,
    get_condition,
    load_graph_spec,
    register_condition,
)
from fleet.graphspec.conditions import register_parametric_condition


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _ScriptedLLM:
    """LLM mock that returns canned text responses, never calls a network."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[Any] = []

    async def complete(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentMessage:
        self.calls.append((list(messages), tools))
        return AgentMessage(role="assistant", content=self._text)

    @property
    def usage(self) -> None:
        return None


def _minimal_spec_dict() -> dict:
    return {
        "version": "0.3",
        "name": "research_team",
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
                    "base_url": None,
                },
                "position": {"x": 100, "y": 40},
            },
            {
                "id": "writer",
                "type": "agent",
                "agent": {
                    "name": "writer",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "system": "Write a one-paragraph summary.",
                    "tools": ["web_search"],
                    "memory_bank": False,
                },
            },
        ],
        "edges": [{"src": "planner", "dst": "writer", "cond": None}],
        "entry": "planner",
        "exit": "writer",
    }


# ---------------------------------------------------------------------------
# schema / round-trip
# ---------------------------------------------------------------------------

def test_spec_roundtrips_through_pydantic():
    raw = _minimal_spec_dict()
    spec = GraphSpec.model_validate(raw)

    assert isinstance(spec, GraphSpec)
    assert spec.name == "research_team"
    assert [n.id for n in spec.nodes] == ["planner", "writer"]
    assert spec.nodes[0].position == Position(x=100, y=40)
    assert spec.nodes[0].agent.tools == []
    assert spec.nodes[1].agent.tools == ["web_search"]

    dumped = spec.model_dump(mode="json")
    spec2 = GraphSpec.model_validate(dumped)
    assert spec2 == spec


def test_spec_accepts_json_string_and_path(tmp_path):
    raw = _minimal_spec_dict()
    text = json.dumps(raw)

    spec_path = tmp_path / "g.json"
    spec_path.write_text(text)

    # We need API keys to construct FleetLLM during load.
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-for-build")

    cg_from_path = load_graph_spec(spec_path)
    cg_from_str = load_graph_spec(text)
    cg_from_dict = load_graph_spec(raw)

    for cg in (cg_from_path, cg_from_str, cg_from_dict):
        assert cg.name == "research_team"
        assert set(cg.nodes) == {"planner", "writer"}
        assert cg.entry == "planner"
        assert cg.exit == "writer"


# ---------------------------------------------------------------------------
# validation errors — must be CLEAR (no eval, no silent passes)
# ---------------------------------------------------------------------------

def test_unknown_provider_raises():
    raw = _minimal_spec_dict()
    raw["nodes"][0]["agent"]["provider"] = "definitely-not-real"
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(raw)
    assert "Unknown provider" in str(exc.value)


def test_unknown_tool_raises():
    raw = _minimal_spec_dict()
    raw["nodes"][0]["agent"]["tools"] = ["no_such_tool_42"]
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(raw)
    assert "Unknown tool" in str(exc.value)


def test_dangling_edge_raises():
    raw = _minimal_spec_dict()
    raw["edges"].append({"src": "planner", "dst": "ghost"})
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(raw)
    assert "unknown node" in str(exc.value)


def test_missing_entry_raises():
    raw = _minimal_spec_dict()
    raw["entry"] = "ghost"
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(raw)
    assert "entry='ghost'" in str(exc.value)


def test_missing_exit_raises():
    raw = _minimal_spec_dict()
    raw["exit"] = "ghost"
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(raw)
    assert "exit='ghost'" in str(exc.value)


def test_duplicate_node_ids_raise():
    raw = _minimal_spec_dict()
    raw["nodes"].append(raw["nodes"][0])
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(raw)
    assert "Duplicate" in str(exc.value)


def test_extra_fields_rejected():
    raw = _minimal_spec_dict()
    raw["nodes"][0]["agent"]["secret"] = "bad"
    with pytest.raises(ValidationError):
        GraphSpec.model_validate(raw)


# ---------------------------------------------------------------------------
# condition registry — names only, no eval
# ---------------------------------------------------------------------------

def test_builtin_conditions_present():
    assert get_condition("always") is not None
    assert get_condition("max_steps_not_hit") is not None
    # parametric form
    assert get_condition("scratchpad_true:my_flag") is not None


def test_builtin_always_and_max_steps_not_hit_evaluate():
    state = GraphState(goal="t")
    assert get_condition("always")(state) is True

    not_hit = get_condition("max_steps_not_hit")
    assert not_hit(state) is True
    hit_state = state.model_copy(update={"metadata": {"terminated_by": "max_steps"}})
    assert not_hit(hit_state) is False


def test_scratchpad_true_resolves_via_key():
    pred = get_condition("scratchpad_true:ready")
    assert pred(GraphState(goal="t")) is False
    assert pred(GraphState(goal="t", scratchpad={"ready": True})) is True
    assert pred(GraphState(goal="t", scratchpad={"ready": ""})) is False


def test_unknown_condition_name_is_rejected_by_loader():
    raw = _minimal_spec_dict()
    raw["edges"][0]["cond"] = "i_was_never_registered"
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(raw)
    # The error must point at the unknown condition name — never eval'd.
    assert "i_was_never_registered" in str(exc.value)


def test_custom_condition_registers_and_loads():
    register_condition("test_custom_true", lambda s: True)
    raw = _minimal_spec_dict()
    raw["edges"][0]["cond"] = "test_custom_true"
    spec = GraphSpec.model_validate(raw)
    assert spec.edges[0].cond == "test_custom_true"


def test_parametric_condition_registers_and_loads():
    def _factory(arg: str):
        def _pred(state: GraphState) -> bool:
            return state.scratchpad.get("x") == arg
        return _pred

    register_parametric_condition("test_eq", _factory)
    pred = get_condition("test_eq:hello")
    assert pred(GraphState(goal="t", scratchpad={"x": "hello"})) is True
    assert pred(GraphState(goal="t", scratchpad={"x": "nope"})) is False


# ---------------------------------------------------------------------------
# end-to-end: load + run with a mocked LLM
# ---------------------------------------------------------------------------

async def test_load_and_run_with_mocked_llm(monkeypatch):
    raw = _minimal_spec_dict()

    # Avoid FleetLLM construction failing on missing keys.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build")

    cg = load_graph_spec(raw)

    # Swap each node's bound LLM for a deterministic mock so we don't hit a
    # real provider. Nodes are agent.step bound methods.
    for name, node in cg.nodes.items():
        agent = node.fn.__self__  # type: ignore[attr-defined]
        agent.llm = _ScriptedLLM(text=f"{name} done")

    state = GraphState(goal="What is the capital of France?")
    final = await cg.run(state)

    # Both nodes ran in sequence (planner → writer). The writer's reply is
    # the last assistant message.
    assistant_msgs = [m for m in final.messages if m.role == "assistant"]
    assert assistant_msgs[-1].content == "writer done"
    assert any(m.content == "planner done" for m in assistant_msgs)


async def test_load_and_run_with_conditional_edge(monkeypatch):
    """A conditional edge that always evaluates False prevents traversal."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build")

    register_condition("test_never", lambda s: False)
    register_condition("test_yes", lambda s: True)

    raw = _minimal_spec_dict()
    raw["nodes"].append({
        "id": "skipped",
        "type": "agent",
        "agent": {
            "name": "skipped",
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "system": "never reached",
            "tools": [],
            "memory_bank": False,
        },
    })
    # Two edges out of planner: one False → skipped, one True → writer.
    raw["edges"] = [
        {"src": "planner", "dst": "skipped", "cond": "test_never"},
        {"src": "planner", "dst": "writer", "cond": "test_yes"},
    ]

    cg = load_graph_spec(raw)
    visited: list[str] = []
    for name, node in cg.nodes.items():
        agent = node.fn.__self__  # type: ignore[attr-defined]
        agent.llm = _ScriptedLLM(text=f"{name} done")
        original_step = agent.step
        # Wrap step so we know which nodes actually executed.
        def _tracker(state, _agent=agent, _name=name, _orig=original_step):
            visited.append(_name)
            return _orig(state)
        node.fn = _tracker  # type: ignore[assignment]

    state = GraphState(goal="hello")
    await cg.run(state)

    assert "planner" in visited
    assert "writer" in visited
    assert "skipped" not in visited


# ---------------------------------------------------------------------------
# CLI: `fleet run graph.json` accepts JSON specs
# ---------------------------------------------------------------------------

def test_cli_run_accepts_json_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-for-build")

    raw = _minimal_spec_dict()
    spec_path = tmp_path / "graph.json"
    spec_path.write_text(json.dumps(raw))

    # Swap FleetLLM at the loader call site so the CLI's load path doesn't
    # need a live provider. Patch on Agent's llm attribute via _build_agent.
    # Easiest: monkey-patch FleetLLM.complete on the class.
    from fleet.providers import client as _client

    async def _fake_complete(self, messages, tools=None):  # noqa: ARG001
        return AgentMessage(role="assistant", content="cli ok")

    monkeypatch.setattr(_client.FleetLLM, "complete", _fake_complete)

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(spec_path), "--goal", "hi"])
    assert result.exit_code == 0, result.output
    assert "done" in result.output
