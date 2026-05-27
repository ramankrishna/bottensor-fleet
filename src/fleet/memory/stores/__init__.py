"""MemoryStore protocol — the abstract interface every backend implements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from fleet.memory.item import MemoryItem


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence layer for MemoryItem records."""

    def add(self, item: MemoryItem) -> None:
        """Insert a new memory item. Item.embedding must be populated."""
        ...

    def get(self, item_id: str) -> MemoryItem | None:
        """Fetch a memory item by id, or None if not found."""
        ...

    def update(self, item: MemoryItem) -> None:
        """Replace an existing memory item. Raises KeyError if not found."""
        ...

    def delete(self, item_id: str) -> None:
        """Remove a memory item. No-op if not present."""
        ...

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """Return up to k (item, similarity) pairs, sorted by descending similarity."""
        ...

    def list_all(self, scope: str | None = None) -> list[MemoryItem]:
        """Return every stored item, optionally filtered by scope."""
        ...
