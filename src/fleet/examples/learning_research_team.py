"""Research team that learns from past runs (planner → 2 researchers → writer).

Each run retrieves memories from a shared ReasoningBank and, on completion,
writes back distilled lessons. Subsequent runs benefit from prior trajectories
(e.g., better search strategies, sources to avoid, common pitfalls).

Run me with:
    fleet run learning_research_team.py --goal "your question here"

Or to scale the writeback contrast:
    fleet run learning_research_team.py --matts 3 --goal "your question here"
"""
import asyncio

from fleet import Agent, Graph
from fleet.agents.agent import AgentMessage
from fleet.core.state import GraphState, append_message, set_scratchpad
from fleet.memory import ReasoningBank
from fleet.providers.client import FleetLLM


# ── Shared bank ───────────────────────────────────────────────────────────────
# One bank, shared across every agent. Persisted to the default SQLite store so
# memories survive across `fleet run` invocations.

_judge      = FleetLLM("anthropic", "claude-sonnet-4-6")
_inducer    = FleetLLM("anthropic", "claude-sonnet-4-6")

bank = ReasoningBank(
    scope="research_team",
    judge_llm=_judge,
    induction_llm=_inducer,
)


# ── Agents (memory-aware) ─────────────────────────────────────────────────────

planner = Agent(
    name="planner",
    goal=(
        "Break the user's goal into exactly TWO sub-questions. "
        "Respond ONLY with a JSON object: "
        '{"q1": "<first sub-question>", "q2": "<second sub-question>"}'
    ),
    model="anthropic/claude-sonnet-4-6",
    tools=[],
    max_iters=1,
    memory_bank=bank,
)

researcher_a = Agent(
    name="researcher_a",
    goal="Answer the sub-question stored in scratchpad['q1'] thoroughly, with sources.",
    model="anthropic/claude-sonnet-4-6",
    tools=["web_search", "web_fetch"],
    memory_bank=bank,
)

researcher_b = Agent(
    name="researcher_b",
    goal="Answer the sub-question stored in scratchpad['q2'] thoroughly, with sources.",
    model="anthropic/claude-sonnet-4-6",
    tools=["web_search", "web_fetch"],
    memory_bank=bank,
)

writer = Agent(
    name="writer",
    goal=(
        "Synthesise the answers from researcher_a and researcher_b "
        "(found in scratchpad) into a single, well-structured report."
    ),
    model="anthropic/claude-sonnet-4-6",
    tools=[],
    max_iters=1,
    memory_bank=bank,
)


# ── Node wrappers ─────────────────────────────────────────────────────────────

async def plan_step(state: GraphState) -> GraphState:
    import json

    state = await planner.step(state)
    last = state.messages[-1]
    try:
        qs = json.loads(last.content or "")
        state = set_scratchpad(state, "q1", qs.get("q1", ""))
        state = set_scratchpad(state, "q2", qs.get("q2", ""))
    except (ValueError, AttributeError):
        goal = state.goal
        state = set_scratchpad(state, "q1", f"Part 1: {goal}")
        state = set_scratchpad(state, "q2", f"Part 2: {goal}")
    return state


async def research_a_step(state: GraphState) -> GraphState:
    state = await researcher_a.step(state)
    answer = state.messages[-1].content if state.messages else ""
    return set_scratchpad(state, "researcher_a_answer", answer)


async def research_b_step(state: GraphState) -> GraphState:
    state = await researcher_b.step(state)
    answer = state.messages[-1].content if state.messages else ""
    return set_scratchpad(state, "researcher_b_answer", answer)


async def write_step(state: GraphState) -> GraphState:
    a_ans = state.scratchpad.get("researcher_a_answer", "(no answer)")
    b_ans = state.scratchpad.get("researcher_b_answer", "(no answer)")
    brief = (
        f"[Researcher A answered]: {a_ans[:800]}\n\n"
        f"[Researcher B answered]: {b_ans[:800]}"
    )
    state = append_message(state, AgentMessage(role="user", content=brief))
    return await writer.step(state)


# ── Graph ─────────────────────────────────────────────────────────────────────

graph = (
    Graph("learning_research_team")
    .add_node("planner",      plan_step)
    .add_node("researcher_a", research_a_step)
    .add_node("researcher_b", research_b_step)
    .add_node("writer",       write_step)
    .add_edge("planner",      "researcher_a")
    .add_edge("planner",      "researcher_b")
    .add_edge("researcher_a", "writer")
    .add_edge("researcher_b", "writer")
    .set_entry("planner")
    .set_exit("writer")
    .compile()
)


if __name__ == "__main__":
    state = GraphState(
        goal="How is AI being used in climate science, and what are the main risks?"
    )
    final = asyncio.run(graph.run(state))
    print(final.messages[-1].content)

    # After the run, distilled memories live in the shared bank.
    items = asyncio.run(bank.list())
    print(f"\n[bank] {len(items)} memory items in scope={bank.scope!r}")
    for it in items[-5:]:
        print(f"  • {it.title}  ({it.source}, conf={it.confidence:.2f})")
