"""Tests for fleet.memory.retrieval.retrieve_relevant."""

from __future__ import annotations

from dataclasses import dataclass, field

from fleet.memory.item import MemoryItem
from fleet.memory.retrieval import retrieve_relevant
from fleet.memory.stores.inmemory import InMemoryStore


class _FixedEmbedder:
    """Returns a pre-set vector for any input."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    @property
    def dim(self) -> int:
        return len(self._vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


@dataclass
class _FakeBank:
    store: InMemoryStore
    embedder: _FixedEmbedder
    merge_thresholds: tuple[float, float] = (0.75, 0.92)
    induction_llm: object | None = None
    scope: str = "global"
    judge_llm: object | None = None
    async_writeback: bool = True


def _make(idx: int, embedding: list[float], **kw) -> MemoryItem:
    return MemoryItem(
        title=f"t{idx}",
        description=f"d{idx}",
        content=f"c{idx}",
        source="success",
        task_signature="sig",
        embedding=embedding,
        **kw,
    )


async def test_retrieve_ranks_by_score_times_confidence() -> None:
    store = InMemoryStore()
    # All have the same similarity to the query (identical vec), so ranking is by confidence.
    a = _make(1, [1.0, 0.0], confidence=0.9)
    b = _make(2, [1.0, 0.0], confidence=0.5)
    c = _make(3, [1.0, 0.0], confidence=0.7)
    for it in (a, b, c):
        store.add(it)

    bank = _FakeBank(store=store, embedder=_FixedEmbedder([1.0, 0.0]))
    results = await retrieve_relevant(bank, "task", k=3)
    assert [it.id for it in results] == [a.id, c.id, b.id]


async def test_retrieve_filters_by_min_confidence() -> None:
    store = InMemoryStore()
    high = _make(1, [1.0, 0.0], confidence=0.8)
    low = _make(2, [1.0, 0.0], confidence=0.1)
    store.add(high)
    store.add(low)

    bank = _FakeBank(store=store, embedder=_FixedEmbedder([1.0, 0.0]))
    results = await retrieve_relevant(bank, "task", k=5, min_confidence=0.3)
    assert [it.id for it in results] == [high.id]


async def test_retrieve_increments_use_count_and_last_used() -> None:
    store = InMemoryStore()
    item = _make(1, [1.0, 0.0], confidence=0.9)
    store.add(item)

    bank = _FakeBank(store=store, embedder=_FixedEmbedder([1.0, 0.0]))
    results = await retrieve_relevant(bank, "task", k=1)
    assert len(results) == 1
    assert results[0].use_count == 1
    assert results[0].last_used_at is not None

    # Persisted to store.
    stored = store.get(item.id)
    assert stored is not None
    assert stored.use_count == 1
    assert stored.last_used_at is not None


async def test_retrieve_respects_k_after_overfetch() -> None:
    store = InMemoryStore()
    for i in range(10):
        store.add(_make(i, [1.0, 0.0], confidence=0.9))
    bank = _FakeBank(store=store, embedder=_FixedEmbedder([1.0, 0.0]))
    results = await retrieve_relevant(bank, "task", k=3)
    assert len(results) == 3


async def test_retrieve_scope_filter() -> None:
    store = InMemoryStore()
    g = _make(1, [1.0, 0.0], scope="global", confidence=0.9)
    a = _make(2, [1.0, 0.0], scope="agent:x", confidence=0.9)
    store.add(g)
    store.add(a)
    bank = _FakeBank(store=store, embedder=_FixedEmbedder([1.0, 0.0]))
    results = await retrieve_relevant(bank, "task", k=5, scope="agent:x")
    assert [it.id for it in results] == [a.id]


async def test_retrieve_empty_when_k_zero() -> None:
    store = InMemoryStore()
    store.add(_make(1, [1.0, 0.0], confidence=0.9))
    bank = _FakeBank(store=store, embedder=_FixedEmbedder([1.0, 0.0]))
    assert await retrieve_relevant(bank, "task", k=0) == []
