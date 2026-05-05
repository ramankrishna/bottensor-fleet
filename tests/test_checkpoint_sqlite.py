import pytest

from fleet.core.checkpoint import SQLiteCheckpoint
from fleet.core.messages import AgentMessage
from fleet.core.state import GraphState, append_message


@pytest.fixture
def checkpoint(tmp_path):
    return SQLiteCheckpoint(path=str(tmp_path / "ckpt.db"))


async def test_save_and_load_roundtrip(checkpoint):
    state = GraphState(goal="hello world", metadata={"token_count": 42})
    await checkpoint.save("run1", state)

    loaded = await checkpoint.load("run1")
    assert loaded is not None
    assert loaded.goal == "hello world"
    assert loaded.metadata["token_count"] == 42


async def test_load_nonexistent_returns_none(checkpoint):
    result = await checkpoint.load("does-not-exist")
    assert result is None


async def test_load_nonexistent_db_returns_none(tmp_path):
    ck = SQLiteCheckpoint(path=str(tmp_path / "missing_dir" / "sub" / "ckpt.db"))
    result = await ck.load("x")
    assert result is None


async def test_list_runs_empty(checkpoint):
    runs = await checkpoint.list_runs()
    assert runs == []


async def test_list_runs_after_saves(checkpoint):
    s = GraphState(goal="t")
    await checkpoint.save("run-a", s)
    await checkpoint.save("run-b", s)

    runs = await checkpoint.list_runs()
    assert "run-a" in runs
    assert "run-b" in runs
    assert len(runs) == 2


async def test_save_overwrites_existing(checkpoint):
    s1 = GraphState(goal="first")
    s2 = GraphState(goal="second")
    await checkpoint.save("r1", s1)
    await checkpoint.save("r1", s2)

    loaded = await checkpoint.load("r1")
    assert loaded is not None
    assert loaded.goal == "second"


async def test_messages_survive_roundtrip(checkpoint):
    state = GraphState(goal="msg test")
    msg = AgentMessage(role="user", content="hello")
    state = append_message(state, msg)
    await checkpoint.save("run-msg", state)

    loaded = await checkpoint.load("run-msg")
    assert loaded is not None
    assert len(loaded.messages) == 1
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == "hello"


async def test_scratchpad_survives_roundtrip(checkpoint):
    state = GraphState(goal="scratch test", scratchpad={"node_a": {"x": 1, "y": [2, 3]}})
    await checkpoint.save("run-scratch", state)

    loaded = await checkpoint.load("run-scratch")
    assert loaded is not None
    assert loaded.scratchpad["node_a"]["x"] == 1
    assert loaded.scratchpad["node_a"]["y"] == [2, 3]
