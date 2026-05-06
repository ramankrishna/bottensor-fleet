from __future__ import annotations

from typing import Any

import polyrt
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from fleet.core.messages import AgentMessage, ToolCall
from fleet.errors import ProviderError

_BACKEND_ALIASES: dict[str, str] = {
    "anthropic": "claude",
    "claude":    "claude",
    "openai":    "openai",
    "gpt":       "openai",
    "ollama":    "ollama",
    "mlx":       "mlx",
}


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
        self.backend = _BACKEND_ALIASES.get(backend.lower(), backend.lower())
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
        # polyrt's agenerate doesn't accept tool definitions — when the caller
        # provides tools, bypass polyrt and call the anthropic SDK directly.
        if tools and self.backend == "claude":
            return await self._complete_anthropic_with_tools(messages, tools)

        polyrt_msgs = [_to_polyrt(m) for m in messages]
        call_kwargs: dict[str, Any] = dict(self._extra)

        response: polyrt.Response | None = None
        try:
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
        except polyrt.BackendError as exc:
            raise ProviderError(
                f"{self.backend}/{self.model}: {exc}"
            ) from exc
        except polyrt.ConfigurationError as exc:
            raise ProviderError(
                f"{self.backend} not configured — set the API key env var ({exc})"
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"{self.backend}/{self.model} unexpected error: {type(exc).__name__}: {exc}"
            ) from exc

        assert response is not None
        self._last_usage = response.usage
        return _from_response(response)

    async def _complete_anthropic_with_tools(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]],
    ) -> AgentMessage:
        from anthropic import AsyncAnthropic, APIError

        system, anth_messages = _to_anthropic_format(messages)
        client = AsyncAnthropic()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": anth_messages,
            "temperature": self._temperature,
            "tools": tools,
        }
        if system is not None:
            kwargs["system"] = system

        try:
            msg = await client.messages.create(**kwargs)
        except APIError as exc:
            raise ProviderError(f"{self.backend}/{self.model}: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )

        return AgentMessage(
            role="assistant",
            content="\n".join(text_parts) or None,
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# format helpers
# ---------------------------------------------------------------------------

def _to_anthropic_format(
    messages: list[AgentMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert fleet messages to anthropic SDK message format.

    Returns (system_prompt, anthropic_messages). System messages are concatenated
    and pulled out, since Anthropic takes them as a top-level kwarg.
    Tool calls/results are encoded as Anthropic content blocks.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue
        if m.role == "tool" and m.tool_results:
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content,
                    **({"is_error": True} if tr.is_error else {}),
                }
                for tr in m.tool_results
            ]
            out.append({"role": "user", "content": blocks})
            continue
        if m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            out.append({"role": m.role, "content": blocks})
            continue
        out.append({"role": m.role, "content": m.content or ""})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


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
