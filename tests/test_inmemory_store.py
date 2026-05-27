"""Tests for fleet.memory.stores.inmemory.InMemoryStore."""

from __future__ import annotations

import pytest

from fleet.memory.item import MemoryItem
from fleet.memory.stores import MemoryStore
from fleet.memory.stores.inmemory import InMemoryStore


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


def test_protocol_conformance() -> None:
    store = InMemoryStore()
    assert isinstance(store, MemoryStore)


def test_add_get_round_trip() -> None:
    store = InMemoryStore()
    item = _make(0, [1.0, 0.0, 0.0])
    store.add(item)
    fetched = store.get(item.id)
    assert fetched is not None
    assert fetched.title == item.title
    assert fetched.embedding == item.embedding


def test_add_requires_embedding() -> None:
    store = InMemoryStore()
    item = MemoryItem(
        title="t", description="d", content="c", source="success", task_signature="s"
    )
    with pytest.raises(ValueError):
        store.add(item)


def test_update_then_get() -> None:
    store = InMemoryStore()
    item = _make(0, [1.0, 0.0, 0.0])
    store.add(item)
    item.title = "updated"
    item.use_count = 3
    store.update(item)
    fetched = store.get(item.id)
    assert fetched is not None
    assert fetched.title == "updated"
    assert fetched.use_count == 3


def test_update_missing_raises() -> None:
    store = InMemoryStore()
    item = _make(0, [1.0, 0.0, 0.0])
    with pytest.raises(KeyError):
        store.update(item)


def test_delete() -> None:
    store = InMemoryStore()
    item = _make(0, [1.0, 0.0, 0.0])
    store.add(item)
    store.delete(item.id)
    assert store.get(item.id) is None
    # Delete is idempotent.
    store.delete(item.id)


def test_search_returns_descending_similarity() -> None:
    store = InMemoryStore()
    a = _make(1, [1.0, 0.0, 0.0])
    b = _make(2, [0.0, 1.0, 0.0])
    c = _make(3, [0.7, 0.7, 0.0])
    for it in (a, b, c):
        store.add(it)

    results = store.search([1.0, 0.0, 0.0], k=3)
    assert [item.id for item, _ in results] == [a.id, c.id, b.id]
    assert results[0][1] == pytest.approx(1.0)
    assert results[0][1] >= results[1][1] >= results[2][1]


def test_search_respects_filters() -> None:
    store = InMemoryStore()
    g = _make(1, [1.0, 0.0], scope="global")
    a = _make(2, [1.0, 0.0], scope="agent:researcher")
    store.add(g)
    store.add(a)
    results = store.search([1.0, 0.0], k=5, filters={"scope": "agent:researcher"})
    assert [item.id for item, _ in results] == [a.id]


def test_list_all_scope_filter() -> None:
    store = InMemoryStore()
    g = _make(1, [1.0, 0.0], scope="global")
    a = _make(2, [0.0, 1.0], scope="agent:x")
    store.add(g)
    store.add(a)
    assert {it.id for it in store.list_all()} == {g.id, a.id}
    assert [it.id for it in store.list_all(scope="agent:x")] == [a.id]


def test_get_returns_copy() -> None:
    store = InMemoryStore()
    item = _make(0, [1.0, 0.0])
    store.add(item)
    fetched = store.get(item.id)
    assert fetched is not None
    fetched.title = "mutated"
    again = store.get(item.id)
    assert again is not None
    assert again.title != "mutated"
