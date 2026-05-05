from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fleet.core.state import GraphState


@dataclass
class Node:
    name: str
    fn: Callable[[GraphState], Awaitable[GraphState]]
    retries: int = 1


@dataclass
class Edge:
    src: str
    dst: str
    cond: Callable[[GraphState], bool] | None = None


class Graph:
    def __init__(self, name: str) -> None:
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._entry: str | None = None
        self._exit: str | None = None

    def add_node(self, name: str, fn: Callable, *, retries: int = 1) -> Graph:
        self._nodes[name] = Node(name=name, fn=fn, retries=retries)
        return self

    def add_edge(
        self,
        src: str,
        dst: str,
        *,
        cond: Callable[[GraphState], bool] | None = None,
    ) -> Graph:
        self._edges.append(Edge(src=src, dst=dst, cond=cond))
        return self

    def set_entry(self, name: str) -> Graph:
        self._entry = name
        return self

    def set_exit(self, name: str) -> Graph:
        self._exit = name
        return self

    def compile(self, backend: Any = None, max_steps: int = 50) -> CompiledGraph:
        if self._entry is None:
            raise ValueError("Entry node not set")
        if self._exit is None:
            raise ValueError("Exit node not set")
        if self._entry not in self._nodes:
            raise ValueError(f"Entry node '{self._entry}' not found in nodes")
        if self._exit not in self._nodes:
            raise ValueError(f"Exit node '{self._exit}' not found in nodes")

        for edge in self._edges:
            if edge.src not in self._nodes:
                raise ValueError(f"Edge source '{edge.src}' not found in nodes")
            if edge.dst not in self._nodes:
                raise ValueError(f"Edge destination '{edge.dst}' not found in nodes")

        # BFS reachability from entry
        reachable: set[str] = set()
        queue = [self._entry]
        while queue:
            node_name = queue.pop()
            if node_name in reachable:
                continue
            reachable.add(node_name)
            for edge in self._edges:
                if edge.src == node_name and edge.dst not in reachable:
                    queue.append(edge.dst)

        orphans = set(self._nodes.keys()) - reachable
        if orphans:
            raise ValueError(f"Orphan nodes (unreachable from entry): {sorted(orphans)}")

        # Resolve checkpoint backend
        if backend is None or isinstance(backend, str):
            from fleet.core.checkpoint import RedisCheckpoint, SQLiteCheckpoint

            backend_name = backend or os.environ.get("FLEET_CHECKPOINT_BACKEND", "sqlite")
            if backend_name == "sqlite":
                checkpoint: Any = SQLiteCheckpoint()
            elif backend_name == "redis":
                checkpoint = RedisCheckpoint()
            else:
                raise ValueError(f"Unknown backend: '{backend_name}'")
        else:
            checkpoint = backend

        return CompiledGraph(
            name=self.name,
            nodes=dict(self._nodes),
            edges=list(self._edges),
            entry=self._entry,
            exit_node=self._exit,
            max_steps=max_steps,
            backend=checkpoint,
        )


class CompiledGraph:
    def __init__(
        self,
        name: str,
        nodes: dict[str, Node],
        edges: list[Edge],
        entry: str,
        exit_node: str,
        max_steps: int,
        backend: Any,
    ) -> None:
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.entry = entry
        self.exit = exit_node
        self.max_steps = max_steps
        self._backend = backend
        self._event_bus: Any = None

    def subscribe(self, callback: Callable) -> None:
        from fleet.core.scheduler import EventBus

        if self._event_bus is None:
            self._event_bus = EventBus()
        self._event_bus.subscribe(callback)

    async def run(self, state: GraphState) -> GraphState:
        from fleet.core.scheduler import EventBus, Scheduler

        if self._event_bus is None:
            self._event_bus = EventBus()
        scheduler = Scheduler(self, self._backend, self._event_bus)
        return await scheduler.run(state)
