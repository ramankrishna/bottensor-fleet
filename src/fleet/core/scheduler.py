from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable

from fleet.core.state import GraphState, merge_metadata

if TYPE_CHECKING:
    from fleet.core.checkpoint import CheckpointBackend
    from fleet.core.graph import CompiledGraph

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Callable] = []

    def subscribe(self, callback: Callable) -> None:
        self._subscribers.append(callback)

    async def emit(self, event: dict[str, Any]) -> None:
        for cb in self._subscribers:
            if asyncio.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)


class Scheduler:
    def __init__(
        self,
        graph: CompiledGraph,
        backend: CheckpointBackend,
        event_bus: EventBus | None = None,
    ) -> None:
        self.graph = graph
        self.backend = backend
        self.event_bus = event_bus or EventBus()

    async def _run_node(self, name: str, state: GraphState) -> GraphState:
        node = self.graph.nodes[name]
        last_exc: Exception | None = None
        for _ in range(max(node.retries, 1)):
            try:
                return await node.fn(state)
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def _get_next_nodes(self, name: str, state: GraphState) -> list[str]:
        return [
            edge.dst
            for edge in self.graph.edges
            if edge.src == name and (edge.cond is None or edge.cond(state))
        ]

    def _merge_states(self, base: GraphState, updates: list[GraphState]) -> GraphState:
        base_count = len(base.messages)
        new_messages = [msg for upd in updates for msg in upd.messages[base_count:]]

        merged_scratch = dict(base.scratchpad)
        for upd in updates:
            merged_scratch.update(upd.scratchpad)

        merged_meta = dict(base.metadata)
        for upd in updates:
            merged_meta.update(upd.metadata)

        return base.model_copy(
            update={
                "messages": [*base.messages, *new_messages],
                "scratchpad": merged_scratch,
                "metadata": merged_meta,
            }
        )

    async def run(self, state: GraphState) -> GraphState:
        run_id: str = state.metadata.get("run_id") or f"run_{uuid.uuid4().hex[:8]}"
        state = merge_metadata(state, {"run_id": run_id})

        current: set[str] = {self.graph.entry}
        steps = 0

        while current and steps < self.graph.max_steps:
            node_list = sorted(current)
            current = set()

            if len(node_list) == 1:
                name = node_list[0]
                state = await self._run_node(name, state)
                steps += 1
                await self.backend.save(run_id, state)
                await self.event_bus.emit({"type": "node_complete", "node": name, "step": steps})
                if name == self.graph.exit:
                    state = merge_metadata(state, {"terminated_by": "exit_node"})
                    break
                current = set(self._get_next_nodes(name, state))
            else:
                # Fan-out: run all concurrently, then merge
                results = await asyncio.gather(*[self._run_node(n, state) for n in node_list])
                state = self._merge_states(state, list(results))
                steps += len(node_list)
                await self.backend.save(run_id, state)
                for name in node_list:
                    await self.event_bus.emit(
                        {"type": "node_complete", "node": name, "step": steps}
                    )
                if self.graph.exit in node_list:
                    state = merge_metadata(state, {"terminated_by": "exit_node"})
                    break
                for name in node_list:
                    current.update(self._get_next_nodes(name, state))
        else:
            if steps >= self.graph.max_steps:
                logger.warning(
                    f"Run {state.metadata.get('run_id', '?')} terminated by "
                    f"max_steps={self.graph.max_steps}. This usually indicates an "
                    "unbounded cycle in the graph."
                )
                state = merge_metadata(state, {"terminated_by": "max_steps"})

        await self._run_memory_writebacks(state)
        return state

    async def _run_memory_writebacks(self, state: GraphState) -> None:
        """Fire ``ingest_trajectory`` on every unique memory bank reachable
        through the graph's agents."""
        banks: dict[int, Any] = {}
        for node in self.graph.nodes.values():
            owner = getattr(node.fn, "__self__", None)
            if owner is None:
                continue
            bank = getattr(owner, "memory_bank", None)
            if bank is None:
                continue
            banks.setdefault(id(bank), bank)

        for bank in banks.values():
            if getattr(bank, "induction_llm", None) is None:
                continue
            if getattr(bank, "judge_llm", None) is None:
                continue
            try:
                coro = bank.ingest_trajectory(
                    list(state.messages),
                    state.goal,
                    outcome=None,
                )
            except Exception as exc:
                logger.warning(f"memory writeback skipped: {exc}")
                continue
            if getattr(bank, "async_writeback", True):
                asyncio.create_task(_safe_await(coro))
            else:
                try:
                    await coro
                except Exception as exc:
                    logger.warning(f"memory writeback failed: {exc}")


async def _safe_await(coro: Any) -> None:
    try:
        await coro
    except Exception as exc:
        logger.warning(f"memory writeback failed: {exc}")
