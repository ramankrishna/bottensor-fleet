import pytest

from fleet.core.checkpoint import SQLiteCheckpoint
from fleet.core.graph import Graph
from fleet.core.state import GraphState


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sqlite(tmp_path):
    return SQLiteCheckpoint(path=str(tmp_path / "sched.db"))


def _make_fn(name: str, log: list[str]):
    async def fn(state: GraphState) -> GraphState:
        log.append(name)
        return state
    return fn


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

async def test_sequential_ordering(tmp_path):
    log: list[str] = []

    g = (
        Graph("seq")
        .add_node("a", _make_fn("a", log))
        .add_node("b", _make_fn("b", log))
        .add_node("c", _make_fn("c", log))
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_entry("a")
        .set_exit("c")
    )
    state = GraphState(goal="seq test")
    await g.compile(backend=_sqlite(tmp_path)).run(state)

    assert log == ["a", "b", "c"]


async def test_parallel_fanout(tmp_path):
    log: list[str] = []

    g = (
        Graph("fan")
        .add_node("start", _make_fn("start", log))
        .add_node("br_a", _make_fn("br_a", log))
        .add_node("br_b", _make_fn("br_b", log))
        .add_node("exit", _make_fn("exit", log))
        .add_edge("start", "br_a")
        .add_edge("start", "br_b")
        .add_edge("br_a", "exit")
        .add_edge("br_b", "exit")
        .set_entry("start")
        .set_exit("exit")
    )
    state = GraphState(goal="fanout test")
    await g.compile(backend=_sqlite(tmp_path)).run(state)

    assert log[0] == "start"
    assert log[-1] == "exit"
    assert set(log[1:3]) == {"br_a", "br_b"}


async def test_conditional_edge_true_branch_only(tmp_path):
    log: list[str] = []

    g = (
        Graph("cond")
        .add_node("start", _make_fn("start", log))
        .add_node("yes", _make_fn("yes", log))
        .add_node("no", _make_fn("no", log))
        .add_node("exit", _make_fn("exit", log))
        .add_edge("start", "yes", cond=lambda s: True)
        .add_edge("start", "no", cond=lambda s: False)
        .add_edge("yes", "exit")
        .add_edge("no", "exit")   # keeps "no" reachable for compile()
        .set_entry("start")
        .set_exit("exit")
    )
    state = GraphState(goal="cond test")
    await g.compile(backend=_sqlite(tmp_path)).run(state)

    assert "yes" in log
    assert "no" not in log
    assert log[-1] == "exit"


async def test_max_steps_stops_cycle(tmp_path):
    log: list[str] = []

    g = (
        Graph("cycle")
        .add_node("a", _make_fn("a", log))
        .add_node("b", _make_fn("b", log))
        .add_node("exit", _make_fn("exit", log))
        .add_edge("a", "b")
        .add_edge("b", "a")
        .add_edge("b", "exit", cond=lambda s: False)  # never taken; keeps exit reachable
        .set_entry("a")
        .set_exit("exit")
    )
    state = GraphState(goal="cycle test")
    await g.compile(backend=_sqlite(tmp_path), max_steps=5).run(state)

    assert len(log) == 5
    assert log == ["a", "b", "a", "b", "a"]


async def test_exit_stops_further_execution(tmp_path):
    log: list[str] = []

    async def exit_fn(state: GraphState) -> GraphState:
        log.append("exit")
        return state

    async def after_exit(state: GraphState) -> GraphState:
        log.append("after_exit")
        return state

    g = (
        Graph("stop")
        .add_node("start", _make_fn("start", log))
        .add_node("exit", exit_fn)
        .add_node("ghost", after_exit)
        .add_edge("start", "exit")
        .add_edge("exit", "ghost")   # exists in graph but scheduler must not follow
        .set_entry("start")
        .set_exit("exit")
    )
    state = GraphState(goal="stop test")
    await g.compile(backend=_sqlite(tmp_path)).run(state)

    assert "after_exit" not in log
    assert log == ["start", "exit"]


async def test_events_emitted(tmp_path):
    events: list[dict] = []

    async def handler(event: dict) -> None:
        events.append(event)

    g = (
        Graph("evt")
        .add_node("a", _make_fn("a", []))
        .add_node("b", _make_fn("b", []))
        .add_edge("a", "b")
        .set_entry("a")
        .set_exit("b")
    )
    cg = g.compile(backend=_sqlite(tmp_path))
    cg.subscribe(handler)

    state = GraphState(goal="event test")
    await cg.run(state)

    node_names = [e["node"] for e in events]
    assert "a" in node_names
    assert "b" in node_names


async def test_state_metadata_has_run_id(tmp_path):
    g = (
        Graph("meta")
        .add_node("a", lambda s: _identity(s))
        .set_entry("a")
        .set_exit("a")
    )
    state = GraphState(goal="meta test")
    result = await g.compile(backend=_sqlite(tmp_path)).run(state)

    assert "run_id" in result.metadata


async def _identity(state: GraphState) -> GraphState:
    return state


async def test_node_retry_on_failure(tmp_path):
    attempts: list[int] = []

    async def flaky(state: GraphState) -> GraphState:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("transient")
        return state

    g = (
        Graph("retry")
        .add_node("flaky", flaky, retries=2)
        .set_entry("flaky")
        .set_exit("flaky")
    )
    state = GraphState(goal="retry test")
    await g.compile(backend=_sqlite(tmp_path)).run(state)

    assert len(attempts) == 2
