"""Memory-Aware Test-Time Scaling — run k parallel rollouts and contrast-distill.

MaTTS runs the same task k times in parallel through the agent (or graph),
then asks a contrast LLM to compare the resulting trajectories and extract
higher-quality memories than any single rollout could produce on its own.
Contrast memories are stored with ``source="matts_contrast"`` and pass through
the bank's normal merge logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from fleet.core.messages import AgentMessage
from fleet.core.state import GraphState
from fleet.memory.item import MemoryItem
from fleet.memory.utils import load_prompt, parse_json_strict, render_trajectory

if TYPE_CHECKING:
    from fleet.memory.bank import ReasoningBank

logger = logging.getLogger(__name__)


async def matts_run(
    agent: Any,
    task: str,
    bank: "ReasoningBank | None" = None,
    k: int = 3,
    *,
    contrast_llm: Any = None,
) -> tuple[list[GraphState], list[MemoryItem]]:
    """Run ``agent`` k times in parallel on ``task`` and distill contrastive memories.

    ``agent`` is anything with an async ``run(GraphState) -> GraphState`` method —
    typically a CompiledGraph wrapping an Agent. If ``bank`` is omitted, the
    function looks for ``memory_bank`` on the runnable itself or on any node
    owner in its graph.

    ``contrast_llm`` defaults to ``bank.induction_llm``. Returns
    ``(rollout_states, distilled_memories)`` — the latter list may be empty if
    the contrast LLM produced no valid items.
    """
    if k < 1:
        raise ValueError(f"matts_run: k must be >= 1, got {k}")

    resolved_bank = bank if bank is not None else _resolve_bank(agent)
    if resolved_bank is None:
        raise ValueError(
            "matts_run: no memory_bank configured — pass bank explicitly or "
            "use an agent/graph whose node owners expose memory_bank."
        )

    llm = contrast_llm if contrast_llm is not None else resolved_bank.induction_llm
    if llm is None:
        raise ValueError(
            "matts_run: no contrast LLM — pass contrast_llm or set "
            "induction_llm on the bank."
        )

    states = await asyncio.gather(*[_run_one(agent, task, i) for i in range(k)])

    distilled = await _contrast_distill(list(states), task, llm, resolved_bank)
    return list(states), distilled


async def _run_one(runnable: Any, task: str, index: int) -> GraphState:
    state = GraphState(goal=task, metadata={"matts_rollout": index})
    return await runnable.run(state)


def _resolve_bank(runnable: Any) -> "ReasoningBank | None":
    bank = getattr(runnable, "memory_bank", None)
    if bank is not None:
        return bank
    nodes = getattr(runnable, "nodes", None)
    if nodes is None:
        return None
    for node in nodes.values():
        owner = getattr(getattr(node, "fn", None), "__self__", None)
        if owner is None:
            continue
        bank = getattr(owner, "memory_bank", None)
        if bank is not None:
            return bank
    return None


async def _contrast_distill(
    states: list[GraphState],
    task: str,
    llm: Any,
    bank: "ReasoningBank",
) -> list[MemoryItem]:
    from fleet.memory.merge import integrate_candidate

    rendered = [
        f"=== Trajectory {i} ===\n{render_trajectory(list(s.messages))}"
        for i, s in enumerate(states, start=1)
    ]
    template = load_prompt("matts_contrast")
    prompt = template.format(task=task, trajectories="\n\n".join(rendered))

    response = await llm.complete([AgentMessage(role="user", content=prompt)])
    raw = response.content or ""
    try:
        payload = parse_json_strict(raw)
    except ValueError as exc:
        hint = (
            " — response appears truncated, consider raising max_tokens on the contrast LLM"
            if _looks_truncated(raw) else ""
        )
        logger.warning(
            "matts_contrast: skipping distillation, could not parse LLM response as JSON%s (%s)",
            hint, exc,
        )
        return []

    if not isinstance(payload, list):
        logger.warning(
            "matts_contrast: skipping distillation, response must be a JSON array, got %s",
            type(payload).__name__,
        )
        return []

    integrated: list[MemoryItem] = []
    for entry in payload[:3]:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "")).strip()
        description = str(entry.get("description", "")).strip()
        content = str(entry.get("content", "")).strip()
        if not (title and description and content):
            continue
        candidate = MemoryItem(
            title=title,
            description=description,
            content=content,
            source="matts_contrast",
            task_signature=task,
            scope=bank.scope,
            embedding=None,
        )
        candidate.embedding = bank.embedder.embed([candidate.signature_text()])[0]
        _action, stored = await integrate_candidate(bank, candidate)
        integrated.append(stored)
    return integrated


def _looks_truncated(text: str) -> bool:
    stripped = text.strip().rstrip("`").rstrip()
    return bool(stripped) and stripped[-1] not in ("}", "]")
