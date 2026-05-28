"""Tests for fleet.memory.stores.sqlite_vec.SQLiteVecStore."""

from __future__ import annotations

from pathlib import Path

import pytest

sqlite_vec = pytest.importorskip("sqlite_vec")

from fleet.memory.item import MemoryItem  # noqa: E402
from fleet.memory.stores import MemoryStore  # noqa: E402
from fleet.memory.stores.sqlite_vec import SQLiteVecStore  # noqa: E402


DIM = 4  # tiny dim so test vectors are readable


def _make(idx: int, embedding: list[float], scope: str = "global") -> MemoryItem:
    return MemoryItem(
        title=f"t{idx}",
        description=f"d{idx}",
        content=f"c{idx}",
        source="success",
        task_signature="sig",
        scope=scope,
        embedding=embedding,
    )


@pytest.fixture
def store(tmp_path: Path):
    s = SQLiteVecStore(db_path=tmp_path / "rb.db", dim=DIM)
    yield s
    s.close()


def test_protocol_conformance(store: SQLiteVecStore) -> None:
    assert isinstance(store, MemoryStore)


def test_add_get_round_trip(store: SQLiteVecStore) -> None:
    item = _make(0, [1.0, 0.0, 0.0, 0.0])
    store.add(item)
    fetched = store.get(item.id)
    assert fetched is not None
    assert fetched.title == item.title
    assert fetched.scope == item.scope
    assert fetched.embedding == item.embedding


def test_add_requires_matching_dim(store: SQLiteVecStore) -> None:
    item = _make(0, [1.0, 0.0])
    with pytest.raises(ValueError):
        store.add(item)


def test_update_changes_fields(store: SQLiteVecStore) -> None:
    item = _make(0, [1.0, 0.0, 0.0, 0.0])
    store.add(item)
    item.title = "renamed"
    item.use_count = 5
    item.embedding = [0.0, 1.0, 0.0, 0.0]
    store.update(item)
    fetched = store.get(item.id)
    assert fetched is not None
    assert fetched.title == "renamed"
    assert fetched.use_count == 5
    assert fetched.embedding == [0.0, 1.0, 0.0, 0.0]


def test_update_missing_raises(store: SQLiteVecStore) -> None:
    with pytest.raises(KeyError):
        store.update(_make(0, [1.0, 0.0, 0.0, 0.0]))


def test_delete(store: SQLiteVecStore) -> None:
    item = _make(0, [1.0, 0.0, 0.0, 0.0])
    store.add(item)
    store.delete(item.id)
    assert store.get(item.id) is None
    store.delete(item.id)  # idempotent


def test_search_returns_descending_similarity(store: SQLiteVecStore) -> None:
    a = _make(1, [1.0, 0.0, 0.0, 0.0])
    b = _make(2, [0.0, 1.0, 0.0, 0.0])
    c = _make(3, [0.7, 0.7, 0.0, 0.0])
    for it in (a, b, c):
        store.add(it)

    results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
    assert [item.id for item, _ in results] == [a.id, c.id, b.id]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_search_respects_filters(store: SQLiteVecStore) -> None:
    g = _make(1, [1.0, 0.0, 0.0, 0.0], scope="global")
    a = _make(2, [1.0, 0.0, 0.0, 0.0], scope="agent:researcher")
    store.add(g)
    store.add(a)
    results = store.search(
        [1.0, 0.0, 0.0, 0.0], k=5, filters={"scope": "agent:researcher"}
    )
    assert [item.id for item, _ in results] == [a.id]


def test_list_all_scope_filter(store: SQLiteVecStore) -> None:
    g = _make(1, [1.0, 0.0, 0.0, 0.0], scope="global")
    a = _make(2, [0.0, 1.0, 0.0, 0.0], scope="agent:x")
    store.add(g)
    store.add(a)
    assert {it.id for it in store.list_all()} == {g.id, a.id}
    assert [it.id for it in store.list_all(scope="agent:x")] == [a.id]


def test_db_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "rb.db"
    s1 = SQLiteVecStore(db_path=path, dim=DIM)
    item = _make(0, [0.1, 0.2, 0.3, 0.4])
    s1.add(item)
    s1.close()

    s2 = SQLiteVecStore(db_path=path, dim=DIM)
    fetched = s2.get(item.id)
    assert fetched is not None
    assert fetched.title == item.title
    s2.close()
