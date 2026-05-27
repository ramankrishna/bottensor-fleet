"""Tests for fleet.memory.item.MemoryItem."""

from __future__ import annotations

from datetime import datetime

from fleet.memory.item import MemoryItem


def test_memory_item_defaults() -> None:
    item = MemoryItem(
        title="Use rg over grep",
        description="ripgrep is faster",
        content="When searching large repos, prefer rg.",
        source="success",
        task_signature="search-code",
    )
    assert item.id.startswith("mem_")
    assert len(item.id) == len("mem_") + 10
    assert item.scope == "global"
    assert item.use_count == 0
    assert item.confidence == 1.0
    assert item.embedding is None
    assert item.last_used_at is None
    assert item.related_ids == []
    assert isinstance(item.created_at, datetime)


def test_memory_item_round_trip() -> None:
    original = MemoryItem(
        title="t",
        description="d",
        content="c",
        source="failure",
        task_signature="sig",
        scope="agent:researcher",
        embedding=[0.1, 0.2, 0.3],
        use_count=3,
        confidence=0.7,
        related_ids=["mem_abc", "mem_def"],
    )
    blob = original.model_dump_json()
    restored = MemoryItem.model_validate_json(blob)
    assert restored == original


def test_signature_text_concatenates_fields() -> None:
    item = MemoryItem(
        title="T",
        description="D",
        content="C",
        source="manual",
        task_signature="x",
    )
    assert item.signature_text() == "T\nD\nC"


def test_unique_ids() -> None:
    a = MemoryItem(title="a", description="", content="", source="success", task_signature="s")
    b = MemoryItem(title="b", description="", content="", source="success", task_signature="s")
    assert a.id != b.id
