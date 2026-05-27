"""Tests for fleet.memory.merge.integrate_candidate (3-branch logic)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from fleet.core.messages import AgentMessage
from fleet.memory.item import MemoryItem
from fleet.memory.merge import integrate_candidate
from fleet.memory.stores.inmemory import InMemoryStore


class _DirectionalEmbedder:
    """Returns the next pre-set vector in sequence (one per embed call).

    Lets a test stage the similarity between candidate and seeded items
    by controlling exactly what vector the candidate gets.
    """

    def __init__(self, queue: list[list[float]]) -> None:
        self._queue = list(queue)

    @property
    def dim(self) -> int:
        return len(self._queue[0]) if self._queue else 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for _ in texts:
            if not self._queue:
                raise AssertionError("embedder ran out of vectors")
            out.append(self._queue.pop(0))
        return out


class _MergeLLM:
    """Returns a canned merge-prompt response."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def complete(self, messages, tools=None):
        return AgentMessage(role="assistant", content=json.dumps(self._payload))

    @property
    def usage(self):
        return None


@dataclass
class _FakeBank:
    store: InMemoryStore
    embedder: Any
    merge_thresholds: tuple[float, float] = (0.75, 0.92)
    induction_llm: Any = None
    scope: str = "global"
    judge_llm: Any = None
    async_writeback: bool = True


def _seed(store: InMemoryStore, idx: int, vec: list[float]) -> MemoryItem:
    item = MemoryItem(
        title=f"existing {idx}",
        description="existing desc",
        content="existing content",
        source="success",
        task_signature="sig",
        embedding=vec,
    )
    store.add(item)
    return item


def _candidate(vec: list[float] | None = None) -> MemoryItem:
    return MemoryItem(
        title="new candidate",
        description="new desc",
        content="new content",
        source="success",
        task_signature="sig",
        embedding=vec,
    )


def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


# ---------------------------------------------------------------------------
# Branch 1: no neighbors / low sim → insert
# ---------------------------------------------------------------------------

async def test_insert_when_store_empty() -> None:
    store = InMemoryStore()
    bank = _FakeBank(store=store, embedder=_DirectionalEmbedder([[1.0, 0.0]]))
    action, item = await integrate_candidate(bank, _candidate())
    assert action == "insert"
    assert store.get(item.id) is not None


async def test_insert_when_similarity_below_link_threshold() -> None:
    store = InMemoryStore()
    _seed(store, 1, [1.0, 0.0])  # orthogonal to candidate → cos sim = 0
    bank = _FakeBank(store=store, embedder=_DirectionalEmbedder([[0.0, 1.0]]))
    action, _ = await integrate_candidate(bank, _candidate())
    assert action == "insert"
    assert len(store.list_all()) == 2


# ---------------------------------------------------------------------------
# Branch 2: link
# ---------------------------------------------------------------------------

async def test_link_when_similarity_in_window() -> None:
    store = InMemoryStore()
    existing = _seed(store, 1, _norm([1.0, 0.0]))
    # cos(theta) ~ 0.85 — solidly inside (0.75, 0.92).
    cand_vec = _norm([0.85, math.sqrt(1 - 0.85 ** 2)])
    bank = _FakeBank(store=store, embedder=_DirectionalEmbedder([cand_vec]))
    action, candidate = await integrate_candidate(bank, _candidate())
    assert action == "link"

    fetched_existing = store.get(existing.id)
    assert fetched_existing is not None
    assert candidate.id in fetched_existing.related_ids
    fetched_cand = store.get(candidate.id)
    assert fetched_cand is not None
    assert existing.id in fetched_cand.related_ids


# ---------------------------------------------------------------------------
# Branch 3: merge (sim > merge_threshold) via LLM
# ---------------------------------------------------------------------------

async def test_merge_action_calls_llm_and_replaces_existing() -> None:
    store = InMemoryStore()
    existing = _seed(store, 1, _norm([1.0, 0.0]))
    # Identical vector → cosine sim = 1.0, comfortably above merge threshold.
    bank = _FakeBank(
        store=store,
        embedder=_DirectionalEmbedder([_norm([1.0, 0.0])]),
        induction_llm=_MergeLLM(
            {
                "action": "merge",
                "merged_title": "merged title",
                "merged_description": "merged desc",
                "merged_content": "merged content",
            }
        ),
    )
    action, item = await integrate_candidate(bank, _candidate())
    assert action == "merge"
    fetched = store.get(existing.id)
    assert fetched is not None
    assert fetched.title == "merged title"
    # Only one item in store — no new row.
    assert len(store.list_all()) == 1


async def test_replace_action() -> None:
    store = InMemoryStore()
    existing = _seed(store, 1, _norm([1.0, 0.0]))
    bank = _FakeBank(
        store=store,
        embedder=_DirectionalEmbedder([_norm([1.0, 0.0])]),
        induction_llm=_MergeLLM(
            {
                "action": "replace",
                "merged_title": "",
                "merged_description": "",
                "merged_content": "",
            }
        ),
    )
    action, _ = await integrate_candidate(bank, _candidate())
    assert action == "replace"
    fetched = store.get(existing.id)
    assert fetched is not None
    assert fetched.title == "new candidate"
    assert fetched.content == "new content"


async def test_keep_both_falls_through_to_link() -> None:
    store = InMemoryStore()
    existing = _seed(store, 1, _norm([1.0, 0.0]))
    bank = _FakeBank(
        store=store,
        embedder=_DirectionalEmbedder([_norm([1.0, 0.0])]),
        induction_llm=_MergeLLM(
            {
                "action": "keep_both",
                "merged_title": "",
                "merged_description": "",
                "merged_content": "",
            }
        ),
    )
    action, candidate = await integrate_candidate(bank, _candidate())
    assert action == "link"
    assert len(store.list_all()) == 2
    fetched_existing = store.get(existing.id)
    assert fetched_existing is not None
    assert candidate.id in fetched_existing.related_ids


async def test_merge_without_llm_replaces_silently() -> None:
    store = InMemoryStore()
    existing = _seed(store, 1, _norm([1.0, 0.0]))
    bank = _FakeBank(
        store=store,
        embedder=_DirectionalEmbedder([_norm([1.0, 0.0])]),
        induction_llm=None,
    )
    action, _ = await integrate_candidate(bank, _candidate())
    assert action == "replace"
    fetched = store.get(existing.id)
    assert fetched is not None
    assert fetched.title == "new candidate"
