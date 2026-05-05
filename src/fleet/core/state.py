from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from fleet.core.messages import AgentMessage


class GraphState(BaseModel):
    messages: list[AgentMessage] = Field(default_factory=list)
    scratchpad: dict[str, Any] = Field(default_factory=dict)
    goal: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def append_message(state: GraphState, msg: AgentMessage) -> GraphState:
    return state.model_copy(update={"messages": [*state.messages, msg]})


def set_scratchpad(state: GraphState, node: str, value: Any) -> GraphState:
    return state.model_copy(update={"scratchpad": {**state.scratchpad, node: value}})


def merge_metadata(state: GraphState, data: dict[str, Any]) -> GraphState:
    return state.model_copy(update={"metadata": {**state.metadata, **data}})
