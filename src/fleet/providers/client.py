from __future__ import annotations

from typing import Any

import polyrt
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from fleet.core.messages import AgentMessage, ToolCall, ToolResult


class FleetLLM:
    """polyrt-backed LLM client with exponential-backoff retries on 429/5xx."""

    def __init__(
        self,
        backend: str,
        model: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.backend = backend
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._extra: dict[str, Any] = kwargs
        self._last_usage: polyrt.Usage | None = None

    @property
    def usage(self) -> polyrt.Usage | None:
        return self._last_usage

    async def complete(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentMessage:
        polyrt_msgs = [_to_polyrt(m) for m in messages]
        call_kwargs: dict[str, Any] = dict(self._extra)
        if tools:
            call_kwargs["tools"] = tools

        response: polyrt.Response | None = None
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(polyrt.BackendError),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(3),
            reraise=True,
        ):
            with attempt:
                response = await polyrt.agenerate(
                    self.backend,
                    model=self.model,
                    messages=polyrt_msgs,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    **call_kwargs,
                )

        assert response is not None
        self._last_usage = response.usage
        return _from_response(response)


# ---------------------------------------------------------------------------
# format helpers
# ---------------------------------------------------------------------------

def _to_polyrt(msg: AgentMessage) -> polyrt.Message:
    if msg.role == "tool" and msg.tool_results:
        tr = msg.tool_results[0]
        return polyrt.Message(role="tool", content=tr.content, tool_call_id=tr.tool_call_id)

    if msg.tool_calls:
        ptc = [
            polyrt.ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
            for tc in msg.tool_calls
        ]
        return polyrt.Message(role=msg.role, content=msg.content or "", tool_calls=ptc)

    return polyrt.Message(role=msg.role, content=msg.content or "")


def _from_response(resp: polyrt.Response) -> AgentMessage:
    fleet_tcs = [
        ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
        for tc in resp.tool_calls
    ]
    return AgentMessage(
        role="assistant",
        content=resp.text or None,
        tool_calls=fleet_tcs,
    )
