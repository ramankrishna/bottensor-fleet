from __future__ import annotations

import json
import os
from typing import Any

import polyrt
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from fleet.core.messages import AgentMessage, ToolCall
from fleet.errors import ProviderError


_ANTHROPIC_BACKENDS = {"anthropic", "claude"}
_OPENAI_BACKENDS = {"openai"}


class FleetLLM:
    """LLM client that supports tool use across Anthropic and OpenAI.

    Without tools, requests go through polyrt for unified backend dispatch.
    With tools, polyrt is bypassed (it forwards unknown kwargs to backend
    constructors, which reject ``tools``) and the native SDK is called
    directly so we can pass the tool schemas through.
    """

    def __init__(
        self,
        backend: str,
        model: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.backend = backend.lower()
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
        if tools:
            if self.backend in _ANTHROPIC_BACKENDS:
                return await self._anthropic_with_tools(messages, tools)
            if self.backend in _OPENAI_BACKENDS:
                return await self._openai_with_tools(messages, tools)
            raise ProviderError(
                f"Tool use is not supported for backend '{self.backend}'. "
                f"Supported backends: anthropic, claude, openai."
            )
        return await self._polyrt_complete(messages)

    # ------------------------------------------------------------------ polyrt

    async def _polyrt_complete(self, messages: list[AgentMessage]) -> AgentMessage:
        polyrt_msgs = [_to_polyrt(m) for m in messages]
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
                        **self._extra,
                    )
        except polyrt.BackendError as exc:
            raise ProviderError(f"{self.backend}/{self.model}: {exc}") from exc
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

    # --------------------------------------------------------------- anthropic

    async def _anthropic_with_tools(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]],
    ) -> AgentMessage:
        try:
            from anthropic import AsyncAnthropic
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "anthropic backend requires the 'anthropic' package; "
                "install bottensor-fleet with the default extras."
            ) from exc

        api_key = self._extra.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError(
                "anthropic backend requires ANTHROPIC_API_KEY env var or api_key argument."
            )
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self._extra.get("base_url"):
            client_kwargs["base_url"] = self._extra["base_url"]
        client = AsyncAnthropic(**client_kwargs)

        system, anth_messages = _to_anthropic_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": anth_messages,
            "tools": tools,
        }
        if system is not None:
            request_kwargs["system"] = system

        try:
            msg = await _retry_async(
                lambda: client.messages.create(**request_kwargs),
                retry_on=(anthropic.APIStatusError, anthropic.APIConnectionError),
            )
        except anthropic.APIError as exc:
            raise ProviderError(f"{self.backend}/{self.model}: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input) if block.input else {},
                    )
                )

        usage = getattr(msg, "usage", None)
        if usage is not None:
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            self._last_usage = polyrt.Usage(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            )

        return AgentMessage(
            role="assistant",
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------ openai

    async def _openai_with_tools(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]],
    ) -> AgentMessage:
        try:
            from openai import AsyncOpenAI
            import openai
        except ImportError as exc:
            raise ProviderError(
                "openai backend requires the 'openai' package; "
                "install bottensor-fleet with the default extras."
            ) from exc

        api_key = self._extra.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError(
                "openai backend requires OPENAI_API_KEY env var or api_key argument."
            )
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self._extra.get("base_url"):
            client_kwargs["base_url"] = self._extra["base_url"]
        client = AsyncOpenAI(**client_kwargs)

        oai_messages = _to_openai_messages(messages)
        oai_tools = [_tool_to_openai_schema(t) for t in tools]

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": oai_messages,
            "tools": oai_tools,
        }

        try:
            completion = await _retry_async(
                lambda: client.chat.completions.create(**request_kwargs),
                retry_on=(openai.APIStatusError, openai.APIConnectionError),
            )
        except openai.OpenAIError as exc:
            raise ProviderError(f"{self.backend}/{self.model}: {exc}") from exc

        choice = completion.choices[0]
        msg = choice.message
        text = msg.content or None
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            args_raw = tc.function.arguments or "{}"
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = getattr(completion, "usage", None)
        if usage is not None:
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            self._last_usage = polyrt.Usage(
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            )

        return AgentMessage(role="assistant", content=text, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

async def _retry_async(
    factory: Any,
    *,
    retry_on: tuple[type[BaseException], ...],
):
    """Run ``factory()`` (an awaitable factory) with exponential-backoff retries."""
    last_exc: BaseException | None = None
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(retry_on),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    ):
        with attempt:
            try:
                return await factory()
            except BaseException as exc:
                last_exc = exc
                raise
    raise last_exc if last_exc else RuntimeError("retry loop exhausted without result")


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


def _to_anthropic_messages(
    messages: list[AgentMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert Fleet messages to Anthropic message dicts (system pulled out)."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            if m.content:
                system_parts.append(m.content)
            continue
        if m.role == "tool":
            blocks: list[dict[str, Any]] = []
            for tr in m.tool_results:
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content,
                }
                if tr.is_error:
                    block["is_error"] = True
                blocks.append(block)
            out.append({"role": "user", "content": blocks})
            continue
        if m.role == "assistant" and m.tool_calls:
            blocks = []
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
            out.append({"role": "assistant", "content": blocks})
            continue
        # plain user/assistant text
        out.append({"role": m.role, "content": m.content or ""})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def _to_openai_messages(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            for tr in m.tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                )
            continue
        if m.role == "assistant" and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
            continue
        out.append({"role": m.role, "content": m.content or ""})
    return out


def _tool_to_openai_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Accept either an Anthropic-style or OpenAI-style tool dict and return OpenAI form."""
    if tool.get("type") == "function" and "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or tool.get("parameters") or {},
        },
    }
