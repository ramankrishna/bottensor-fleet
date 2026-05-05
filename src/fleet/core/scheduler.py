from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Callable

from fleet.core.state import GraphState, merge_metadata

if TYPE_CHECKING:
    from fleet.core.checkpoint import CheckpointBackend
    from fleet.core.graph import CompiledGraph


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
                state = merge_metadata(state, {"terminated_by": "max_steps"})

        return state
