"""Tests for fleet.memory.judge.judge_trajectory."""

from __future__ import annotations

import json
from typing import Any

import pytest

from fleet.core.messages import AgentMessage
from fleet.memory.judge import judge_trajectory


class _MockLLM:
    def __init__(self, payload: Any, *, fence: bool = False) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        if fence:
            text = f"```json\n{text}\n```"
        self._text = text

    async def complete(self, messages, tools=None):
        return AgentMessage(role="assistant", content=self._text)

    @property
    def usage(self):
        return None


async def test_judge_success() -> None:
    llm = _MockLLM({"outcome": "success", "rationale": "got it done"})
    outcome, rationale = await judge_trajectory([], "t", llm)
    assert outcome == "success"
    assert rationale == "got it done"


async def test_judge_failure() -> None:
    llm = _MockLLM({"outcome": "failure", "rationale": "hit the limit"})
    outcome, rationale = await judge_trajectory([], "t", llm)
    assert outcome == "failure"
    assert rationale == "hit the limit"


async def test_judge_handles_fenced_response() -> None:
    llm = _MockLLM({"outcome": "success", "rationale": "ok"}, fence=True)
    outcome, _ = await judge_trajectory([], "t", llm)
    assert outcome == "success"


async def test_judge_case_insensitive() -> None:
    llm = _MockLLM({"outcome": "SUCCESS", "rationale": "x"})
    outcome, _ = await judge_trajectory([], "t", llm)
    assert outcome == "success"


async def test_judge_invalid_outcome_raises() -> None:
    llm = _MockLLM({"outcome": "maybe", "rationale": "x"})
    with pytest.raises(ValueError):
        await judge_trajectory([], "t", llm)


async def test_judge_non_object_raises() -> None:
    llm = _MockLLM(["not", "an", "object"])
    with pytest.raises(ValueError):
        await judge_trajectory([], "t", llm)


async def test_judge_bad_json_raises() -> None:
    llm = _MockLLM("totally not json")
    with pytest.raises(ValueError):
        await judge_trajectory([], "t", llm)
