"""Utilities for the memory subsystem."""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE | re.MULTILINE)


def parse_json_strict(text: str) -> Any:
    """Parse JSON from an LLM response.

    Strips ```json ... ``` fences if present, then parses. Raises
    ValueError with the original text if parsing fails.
    """
    if not text:
        raise ValueError("Empty LLM response — expected JSON")
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response was not valid JSON: {exc}\n--- raw response ---\n{text}"
        ) from exc


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    from pathlib import Path

    path = Path(__file__).parent / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def render_trajectory(trajectory: list[Any]) -> str:
    """Render a trajectory (list of dicts or AgentMessage objects) as readable text."""
    lines: list[str] = []
    for msg in trajectory:
        if hasattr(msg, "role"):  # AgentMessage
            role = msg.role
            content = msg.content or ""
            tool_calls = getattr(msg, "tool_calls", None) or []
            tool_results = getattr(msg, "tool_results", None) or []
        elif isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls") or []
            tool_results = msg.get("tool_results") or []
        else:
            lines.append(f"[unknown] {msg!r}")
            continue

        if content:
            lines.append(f"[{role}] {content}")
        for tc in tool_calls:
            name = getattr(tc, "name", None) if not isinstance(tc, dict) else tc.get("name")
            args = getattr(tc, "arguments", None) if not isinstance(tc, dict) else tc.get("arguments")
            lines.append(f"[{role}] tool_call: {name}({args})")
        for tr in tool_results:
            content_val = (
                getattr(tr, "content", None)
                if not isinstance(tr, dict)
                else tr.get("content")
            )
            is_error = (
                getattr(tr, "is_error", False)
                if not isinstance(tr, dict)
                else tr.get("is_error", False)
            )
            prefix = "tool_error" if is_error else "tool_result"
            lines.append(f"[{role}] {prefix}: {content_val}")
    return "\n".join(lines)
