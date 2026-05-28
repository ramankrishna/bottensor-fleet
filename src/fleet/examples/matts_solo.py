"""Single agent that self-improves via Memory-Aware Test-Time Scaling (MaTTS).

MaTTS runs the same task k times in parallel and contrast-distills higher-quality
memories than any single rollout could produce. The agent shares one
ReasoningBank, so each invocation benefits from prior contrast memories.

Run me with:
    fleet run matts_solo.py --matts 3 --goal "your question here"

A normal (non-MaTTS) run also works — memories will then be ingested from the
single trajectory via the scheduler's writeback hook:
    fleet run matts_solo.py --goal "your question here"
"""
import asyncio

from fleet import Agent, Graph
from fleet.core.state import GraphState
from fleet.memory import ReasoningBank
from fleet.providers.client import FleetLLM


# ── Shared bank ───────────────────────────────────────────────────────────────

_judge   = FleetLLM("anthropic", "claude-sonnet-4-6")
_inducer = FleetLLM("anthropic", "claude-sonnet-4-6")

bank = ReasoningBank(
    scope="matts_solo",
    judge_llm=_judge,
    induction_llm=_inducer,
)


# ── Agent (memory-aware) ──────────────────────────────────────────────────────

researcher = Agent(
    name="researcher",
    goal=(
        "Answer the user's question thoroughly. Cite specific sources, "
        "and explicitly call out any reasoning steps you find tricky."
    ),
    model="anthropic/claude-sonnet-4-6",
    tools=["web_search", "web_fetch"],
    memory_bank=bank,
    memory_k=5,
)


# ── Graph ─────────────────────────────────────────────────────────────────────

graph = (
    Graph("matts_solo")
    .add_node("researcher", researcher.step)
    .set_entry("researcher")
    .set_exit("researcher")
    .compile()
)


if __name__ == "__main__":
    from fleet.memory.matts import matts_run

    # MaTTS: 3 parallel rollouts, contrast-distill memories from their differences.
    goal = "What are the most common pitfalls when fine-tuning small language models?"
    states, distilled = asyncio.run(matts_run(graph, goal, bank=bank, k=3))

    print(f"\n[matts] {len(states)} rollouts, {len(distilled)} contrast memories distilled:")
    for it in distilled:
        print(f"  • {it.title}  (conf={it.confidence:.2f})")
        print(f"      {it.description}")

    # Show the first rollout's final answer for illustration.
    if states and states[0].messages:
        print("\n--- rollout 0 final answer ---")
        print(states[0].messages[-1].content)
