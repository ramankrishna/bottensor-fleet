"""Distill generalizable memory items from agent trajectories."""

from __future__ import annotations

from typing import Any, Literal

from fleet.core.messages import AgentMessage
from fleet.memory.item import MemoryItem
from fleet.memory.utils import load_prompt, parse_json_strict, render_trajectory


async def distill_memories(
    trajectory: list[AgentMessage | dict[str, Any]],
    task: str,
    outcome: Literal["success", "failure"],
    llm: Any,
    *,
    scope: str = "global",
) -> list[MemoryItem]:
    """Use an LLM to extract 1-3 generalizable MemoryItem records from a trajectory.

    The returned items have embedding=None — the bank fills them in before storage.
    """
    if outcome not in ("success", "failure"):
        raise ValueError(f"outcome must be 'success' or 'failure', got {outcome!r}")

    template_name = "induction_success" if outcome == "success" else "induction_failure"
    template = load_prompt(template_name)
    rendered = template.format(task=task, trajectory=render_trajectory(trajectory))

    message = AgentMessage(role="user", content=rendered)
    response = await llm.complete([message])
    raw = response.content or ""
    payload = parse_json_strict(raw)

    if not isinstance(payload, list):
        raise ValueError(
            f"induction response must be a JSON array, got {type(payload).__name__}"
        )

    items: list[MemoryItem] = []
    for entry in payload[:3]:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "")).strip()
        description = str(entry.get("description", "")).strip()
        content = str(entry.get("content", "")).strip()
        if not (title and description and content):
            continue
        items.append(
            MemoryItem(
                title=title,
                description=description,
                content=content,
                source=outcome,
                task_signature=task,
                scope=scope,
                embedding=None,
            )
        )
    return items
