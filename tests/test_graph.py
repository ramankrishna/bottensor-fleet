import pytest

from fleet.core.graph import Graph
from fleet.core.state import GraphState


async def _noop(state: GraphState) -> GraphState:
    return state


def _make_graph(entry: str = "a", exit_: str = "b") -> Graph:
    g = Graph("test")
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge("a", "b")
    g.set_entry(entry)
    g.set_exit(exit_)
    return g


def test_compile_valid(tmp_path):
    cg = _make_graph().compile(backend=_sqlite(tmp_path))
    assert cg is not None
    assert cg.entry == "a"
    assert cg.exit == "b"
    assert cg.max_steps == 50


def test_compile_custom_max_steps(tmp_path):
    cg = _make_graph().compile(backend=_sqlite(tmp_path), max_steps=10)
    assert cg.max_steps == 10


def test_compile_no_entry(tmp_path):
    g = Graph("t")
    g.add_node("a", _noop)
    g.set_exit("a")
    with pytest.raises(ValueError, match="Entry"):
        g.compile(backend=_sqlite(tmp_path))


def test_compile_no_exit(tmp_path):
    g = Graph("t")
    g.add_node("a", _noop)
    g.set_entry("a")
    with pytest.raises(ValueError, match="Exit"):
        g.compile(backend=_sqlite(tmp_path))


def test_compile_entry_not_in_nodes(tmp_path):
    g = Graph("t")
    g.add_node("a", _noop)
    g.set_entry("missing")
    g.set_exit("a")
    with pytest.raises(ValueError, match="Entry"):
        g.compile(backend=_sqlite(tmp_path))


def test_compile_exit_not_in_nodes(tmp_path):
    g = Graph("t")
    g.add_node("a", _noop)
    g.set_entry("a")
    g.set_exit("missing")
    with pytest.raises(ValueError, match="Exit"):
        g.compile(backend=_sqlite(tmp_path))


def test_compile_invalid_edge_src(tmp_path):
    g = Graph("t")
    g.add_node("a", _noop)
    g.add_edge("ghost", "a")
    g.set_entry("a")
    g.set_exit("a")
    with pytest.raises(ValueError):
        g.compile(backend=_sqlite(tmp_path))


def test_compile_invalid_edge_dst(tmp_path):
    g = Graph("t")
    g.add_node("a", _noop)
    g.add_edge("a", "ghost")
    g.set_entry("a")
    g.set_exit("a")
    with pytest.raises(ValueError):
        g.compile(backend=_sqlite(tmp_path))


def test_compile_orphan_detected(tmp_path):
    g = Graph("t")
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_node("orphan", _noop)
    g.add_edge("a", "b")
    g.set_entry("a")
    g.set_exit("b")
    with pytest.raises(ValueError, match="[Oo]rphan"):
        g.compile(backend=_sqlite(tmp_path))


def test_compile_cycle_allowed(tmp_path):
    g = Graph("cycle")
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_node("exit", _noop)
    g.add_edge("a", "b")
    g.add_edge("b", "a", cond=lambda s: False)   # cycle guarded by condition
    g.add_edge("b", "exit", cond=lambda s: True)
    g.set_entry("a")
    g.set_exit("exit")
    cg = g.compile(backend=_sqlite(tmp_path))
    assert cg is not None


def test_graph_builder_chaining(tmp_path):
    # Verify all builder methods return self for chaining
    g = (
        Graph("chain")
        .add_node("a", _noop)
        .add_node("b", _noop)
        .add_edge("a", "b")
        .set_entry("a")
        .set_exit("b")
    )
    cg = g.compile(backend=_sqlite(tmp_path))
    assert cg is not None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sqlite(tmp_path):
    from fleet.core.checkpoint import SQLiteCheckpoint
    return SQLiteCheckpoint(path=str(tmp_path / "test.db"))
