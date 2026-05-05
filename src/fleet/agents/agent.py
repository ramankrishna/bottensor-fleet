from __future__ import annotations

from typing import Any, Protocol

from fleet.core.messages import AgentMessage, ToolResult
from fleet.core.state import GraphState, append_message
from fleet.tools.base import get_tool, to_anthropic


class LLMProtocol(Protocol):
    async def complete(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentMessage: ...

    @property
    def usage(self) -> Any: ...


class Agent:
    """ReAct agent: repeatedly calls the LLM and dispatches tool calls until
    the model returns a plain text response or *max_iters* is reached."""

    def __init__(
        self,
        llm: LLMProtocol,
        tools: list[str] | None = None,
        max_iters: int = 10,
    ) -> None:
        self.llm = llm
        self.tools: list[str] = tools or []
        self.max_iters = max_iters

    async def step(self, state: GraphState) -> GraphState:
        tool_schemas = [to_anthropic(t) for t in self.tools if get_tool(t) is not None]

        for _ in range(self.max_iters):
            response = await self.llm.complete(
                state.messages,
                tools=tool_schemas or None,
            )
            state = append_message(state, response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                result_msg = await self._dispatch(tc.id, tc.name, tc.arguments)
                state = append_message(state, result_msg)

        return state

    async def _dispatch(
        self, call_id: str, name: str, arguments: dict[str, Any]
    ) -> AgentMessage:
        entry = get_tool(name)
        if entry is None:
            content = f"Error: tool '{name}' is not registered"
            is_error = True
        else:
            try:
                result = await entry["fn"](**arguments)
                content = str(result)
                is_error = False
            except Exception as exc:
                content = f"Error: {exc}"
                is_error = True

        return AgentMessage(
            role="tool",
            tool_results=[ToolResult(tool_call_id=call_id, content=content, is_error=is_error)],
        )
