"""Tests for fleet.memory.induction.distill_memories."""

from __future__ import annotations

import json
from typing import Any

import pytest

from fleet.core.messages import AgentMessage
from fleet.memory.induction import distill_memories


class _MockLLM:
    def __init__(self, payload: Any, *, fence: bool = False) -> None:
        if isinstance(payload, str):
            text = payload
        else:
            text = json.dumps(payload)
        if fence:
            text = f"```json\n{text}\n```"
        self._text = text
        self.calls: list[list[AgentMessage]] = []

    async def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        return AgentMessage(role="assistant", content=self._text)

    @property
    def usage(self):
        return None


def _canned(n: int) -> list[dict[str, str]]:
    return [
        {
            "title": f"title {i}",
            "description": f"description {i}",
            "content": f"content {i}",
        }
        for i in range(n)
    ]


async def test_distill_success_returns_items() -> None:
    llm = _MockLLM(_canned(2))
    items = await distill_memories(
        trajectory=[AgentMessage(role="user", content="hello")],
        task="do the thing",
        outcome="success",
        llm=llm,
    )
    assert len(items) == 2
    assert items[0].title == "title 0"
    assert items[0].source == "success"
    assert items[0].task_signature == "do the thing"
    assert items[0].embedding is None


async def test_distill_failure_marks_source() -> None:
    llm = _MockLLM(_canned(1))
    items = await distill_memories(
        trajectory=[],
        task="t",
        outcome="failure",
        llm=llm,
    )
    assert items[0].source == "failure"


async def test_distill_caps_at_three() -> None:
    llm = _MockLLM(_canned(5))
    items = await distill_memories([], "t", "success", llm)
    assert len(items) == 3


async def test_distill_strips_json_fences() -> None:
    llm = _MockLLM(_canned(1), fence=True)
    items = await distill_memories([], "t", "success", llm)
    assert len(items) == 1


async def test_distill_skips_empty_fields() -> None:
    payload = [
        {"title": "good", "description": "d", "content": "c"},
        {"title": "", "description": "d", "content": "c"},
        {"title": "ok", "description": "d", "content": "c"},
    ]
    llm = _MockLLM(payload)
    items = await distill_memories([], "t", "success", llm)
    assert [it.title for it in items] == ["good", "ok"]


async def test_distill_invalid_outcome() -> None:
    llm = _MockLLM([])
    with pytest.raises(ValueError):
        await distill_memories([], "t", "weird", llm)  # type: ignore[arg-type]


async def test_distill_bad_json_raises() -> None:
    llm = _MockLLM("not json at all {")
    with pytest.raises(ValueError):
        await distill_memories([], "t", "success", llm)


async def test_distill_object_not_array_raises() -> None:
    llm = _MockLLM({"not": "an array"})
    with pytest.raises(ValueError):
        await distill_memories([], "t", "success", llm)
