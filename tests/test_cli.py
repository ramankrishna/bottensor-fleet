"""Tests for the fleet CLI commands."""
from __future__ import annotations

import asyncio

import pytest
from typer.testing import CliRunner

from fleet.cli import app
from fleet.core.checkpoint import SQLiteCheckpoint
from fleet.core.state import GraphState

runner = CliRunner()


_REPLAY_GRAPH_SRC = '''
from fleet import Graph
from fleet.core.state import GraphState


async def step(state: GraphState) -> GraphState:
    return state.model_copy(update={"scratchpad": {**state.scratchpad, "ran": True}})


graph = (
    Graph("replay-test")
    .add_node("step", step)
    .set_entry("step")
    .set_exit("step")
    .compile()
)
'''


@pytest.fixture
def patched_ckpt(tmp_path, monkeypatch):
    """Pin SQLiteCheckpoint to a per-test temp DB everywhere it's imported."""
    db_path = tmp_path / "ckpt.db"

    class _Pinned(SQLiteCheckpoint):
        def __init__(self, path: str | None = None) -> None:
            super().__init__(path=str(db_path))

    monkeypatch.setattr("fleet.core.checkpoint.SQLiteCheckpoint", _Pinned)
    return db_path


def _write_graph(tmp_path):
    p = tmp_path / "replay_graph.py"
    p.write_text(_REPLAY_GRAPH_SRC)
    return p


def _seed_checkpoint(db_path, run_id: str, graph_path: str | None, goal: str = "hello") -> None:
    metadata: dict = {"run_id": run_id}
    if graph_path is not None:
        metadata["graph_source"] = graph_path
    state = GraphState(goal=goal, metadata=metadata)
    asyncio.run(SQLiteCheckpoint(path=str(db_path)).save(run_id, state))


def test_replay_success(tmp_path, patched_ckpt):
    graph_file = _write_graph(tmp_path)
    _seed_checkpoint(patched_ckpt, "run_abc", str(graph_file))

    result = runner.invoke(app, ["replay", "run_abc"])
    assert result.exit_code == 0, result.output
    assert "Replaying run_abc" in result.output
    assert "Replay complete" in result.output


def test_replay_missing_run(patched_ckpt):
    result = runner.invoke(app, ["replay", "nope"])
    assert result.exit_code == 1
    assert "Run not found" in result.output


def test_replay_with_goal_override(tmp_path, patched_ckpt):
    graph_file = _write_graph(tmp_path)
    _seed_checkpoint(patched_ckpt, "run_xyz", str(graph_file), goal="original goal")

    result = runner.invoke(app, ["replay", "run_xyz", "--goal", "new goal"])
    assert result.exit_code == 0, result.output
    assert "new goal" in result.output


def test_replay_missing_graph_source(patched_ckpt):
    _seed_checkpoint(patched_ckpt, "run_old", graph_path=None)

    result = runner.invoke(app, ["replay", "run_old"])
    assert result.exit_code == 1
    assert "graph_source" in result.output


def test_examples_list_includes_solo_agent():
    result = runner.invoke(app, ["examples"])
    assert result.exit_code == 0, result.output
    assert "solo_agent.py" in result.output


def test_examples_extract(tmp_path):
    result = runner.invoke(app, ["examples", "solo_agent", "--dest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "solo_agent.py").exists()
