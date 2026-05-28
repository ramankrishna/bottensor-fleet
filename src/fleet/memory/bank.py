"""ReasoningBank — public API for experience-augmented agent memory.

A bank owns three collaborators:
- a MemoryStore (persistence + vector search)
- an Embedder (text → vector)
- optional LLMs for judging trajectories and distilling/merging memories

The two main flows are ``retrieve(task)`` — pull the top-k useful memories for a
new task — and ``ingest_trajectory(trajectory, task)`` — extract memories from a
completed run and integrate them into the store.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fleet.memory.item import MemoryItem

if TYPE_CHECKING:
    from fleet.memory.embedders import Embedder
    from fleet.memory.stores import MemoryStore

logger = logging.getLogger(__name__)


def _default_store() -> "MemoryStore":
    try:
        from fleet.memory.stores.sqlite_vec import SQLiteVecStore

        return SQLiteVecStore()
    except Exception as exc:  # pragma: no cover - depends on optional extra
        warnings.warn(
            f"SQLiteVecStore unavailable ({type(exc).__name__}: {exc}); "
            "falling back to InMemoryStore (NOT persistent).",
            stacklevel=3,
        )
        from fleet.memory.stores.inmemory import InMemoryStore

        return InMemoryStore()


def _default_embedder() -> "Embedder":
    try:
        from fleet.memory.embedders.minilm import MiniLMEmbedder

        return MiniLMEmbedder()
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "No default embedder available — install sentence-transformers "
            "or pass an explicit embedder to ReasoningBank()."
        ) from exc


class ReasoningBank:
    """High-level API for the agent's experience memory."""

    def __init__(
        self,
        store: "MemoryStore | None" = None,
        embedder: "Embedder | None" = None,
        scope: str = "global",
        judge_llm: Any = None,
        induction_llm: Any = None,
        merge_thresholds: tuple[float, float] = (0.75, 0.92),
        async_writeback: bool = True,
    ) -> None:
        self.store = store if store is not None else _default_store()
        self.embedder = embedder if embedder is not None else _default_embedder()
        self.scope = scope
        self.judge_llm = judge_llm
        self.induction_llm = induction_llm
        self.merge_thresholds = merge_thresholds
        self.async_writeback = async_writeback

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        task: str,
        k: int = 5,
        *,
        scope: str | None = None,
        min_confidence: float = 0.3,
    ) -> list[MemoryItem]:
        from fleet.memory.retrieval import retrieve_relevant

        return await retrieve_relevant(
            self,
            task=task,
            k=k,
            scope=scope if scope is not None else self.scope,
            min_confidence=min_confidence,
        )

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    async def ingest_trajectory(
        self,
        trajectory: list[Any],
        task: str,
        outcome: Literal["success", "failure"] | None = None,
    ) -> list[MemoryItem]:
        """Judge, distill, embed, and integrate memories from one trajectory.

        Returns the list of MemoryItems that were inserted/merged/linked.
        """
        from fleet.memory.induction import distill_memories
        from fleet.memory.judge import judge_trajectory
        from fleet.memory.merge import integrate_candidate

        if not trajectory:
            return []

        resolved_outcome: Literal["success", "failure"]
        if outcome is None:
            if self.judge_llm is None:
                raise ValueError(
                    "ingest_trajectory: outcome=None requires judge_llm on the bank."
                )
            resolved_outcome, _rationale = await judge_trajectory(
                trajectory, task, self.judge_llm
            )
        else:
            resolved_outcome = outcome

        induction_llm = self.induction_llm
        if induction_llm is None:
            raise ValueError(
                "ingest_trajectory: induction_llm must be set on the bank."
            )

        candidates = await distill_memories(
            trajectory=trajectory,
            task=task,
            outcome=resolved_outcome,
            llm=induction_llm,
            scope=self.scope,
        )

        integrated: list[MemoryItem] = []
        for candidate in candidates:
            if candidate.embedding is None:
                candidate.embedding = self.embedder.embed([candidate.signature_text()])[0]
            _action, stored = await integrate_candidate(self, candidate)
            integrated.append(stored)
        return integrated

    # ------------------------------------------------------------------
    # direct CRUD-ish helpers
    # ------------------------------------------------------------------

    async def add_manual(
        self,
        title: str,
        description: str,
        content: str,
        *,
        scope: str | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            title=title,
            description=description,
            content=content,
            source="manual",
            task_signature="",
            scope=scope if scope is not None else self.scope,
        )
        item.embedding = self.embedder.embed([item.signature_text()])[0]
        self.store.add(item)
        return item

    async def list(self, scope: str | None = None) -> list[MemoryItem]:
        return self.store.list_all(scope=scope)

    async def delete(self, id: str) -> None:
        self.store.delete(id)

    # ------------------------------------------------------------------
    # export / import
    # ------------------------------------------------------------------

    async def export(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in self.store.list_all():
                f.write(item.model_dump_json() + "\n")

    async def import_(self, path: str | Path) -> int:
        path = Path(path)
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                item = MemoryItem.model_validate(data)
                if item.embedding is None:
                    item.embedding = self.embedder.embed([item.signature_text()])[0]
                try:
                    self.store.add(item)
                    count += 1
                except ValueError:
                    # duplicate id — overwrite instead
                    self.store.update(item)
                    count += 1
        return count

    # ------------------------------------------------------------------
    # writeback used by the scheduler
    # ------------------------------------------------------------------

    def schedule_writeback(
        self,
        trajectory: "builtins.list[Any]",
        task: str,
        outcome: Literal["success", "failure"] | None = None,
    ) -> "asyncio.Task[builtins.list[MemoryItem]] | builtins.list[MemoryItem] | None":
        """Kick off ``ingest_trajectory``.

        If ``async_writeback`` is True (and an event loop is running), schedule
        it as a background task and return the Task handle. Otherwise run it
        synchronously via ``asyncio.run`` and return the resulting items.

        Returns None if neither judge nor induction LLM is configured (silent
        no-op so the bank can still be used purely for retrieval).
        """
        if self.induction_llm is None and self.judge_llm is None:
            return None

        coro = self.ingest_trajectory(trajectory, task, outcome=outcome)
        if self.async_writeback:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                return loop.create_task(coro)
            return asyncio.run(coro)
        return asyncio.run(coro)
