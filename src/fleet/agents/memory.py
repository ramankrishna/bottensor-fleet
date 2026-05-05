from __future__ import annotations

from typing import Any

from fleet.core.state import GraphState, set_scratchpad


class ScratchpadMemory:
    """Thin dict-backed namespace inside GraphState.scratchpad[node]."""

    def __init__(self, state: GraphState, node: str) -> None:
        self._state = state
        self._node = node

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.scratchpad.get(self._node, {}).get(key, default)

    def set(self, key: str, value: Any) -> GraphState:
        ns = dict(self._state.scratchpad.get(self._node, {}))
        ns[key] = value
        new_state = set_scratchpad(self._state, self._node, ns)
        self._state = new_state
        return new_state

    def as_dict(self) -> dict[str, Any]:
        return dict(self._state.scratchpad.get(self._node, {}))


class VectorMemory:
    """Stub vector memory — requires a vector store backend (not yet wired)."""

    async def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError(
            "VectorMemory requires a vector store backend. "
            "Pass a configured backend via VectorMemory(backend=...)."
        )

    async def search(self, query: str, k: int = 5) -> list[str]:
        raise NotImplementedError(
            "VectorMemory requires a vector store backend."
        )
