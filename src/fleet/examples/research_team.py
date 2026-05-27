"""3-node DAG: planner → [researcher_a, researcher_b] → writer.

Demonstrates parallel fan-out and merge.  The planner decomposes the goal
into two sub-questions (stored in scratchpad), each researcher handles one
in parallel, and the writer synthesises both into a final report.
"""
import asyncio

from fleet import Agent, Graph
from fleet.core.state import GraphState, append_message, set_scratchpad
from fleet.agents.agent import AgentMessage


# ── Nodes ──────────────────────────────────────────────────────────────────────

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
)

researcher_a = Agent(
    name="researcher_a",
    goal="Answer the sub-question stored in scratchpad['q1'] thoroughly, with sources.",
    model="anthropic/claude-sonnet-4-6",
    tools=["web_search", "web_fetch"],
)

researcher_b = Agent(
    name="researcher_b",
    goal="Answer the sub-question stored in scratchpad['q2'] thoroughly, with sources.",
    model="anthropic/claude-sonnet-4-6",
    tools=["web_search", "web_fetch"],
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
)


# ── Planner wrapper: parse JSON and stash sub-questions in scratchpad ──────────

async def plan_step(state: GraphState) -> GraphState:
    import json

    state = await planner.step(state)
    last = state.messages[-1]
    try:
        qs = json.loads(last.content)
        state = set_scratchpad(state, "q1", qs.get("q1", ""))
        state = set_scratchpad(state, "q2", qs.get("q2", ""))
    except (ValueError, AttributeError):
        # If the model didn't return clean JSON, fall back gracefully.
        goal = state.goal
        state = set_scratchpad(state, "q1", f"Part 1: {goal}")
        state = set_scratchpad(state, "q2", f"Part 2: {goal}")
    return state


# ── Writer wrapper: inject researcher answers into context first ───────────────

async def write_step(state: GraphState) -> GraphState:
    a_ans = state.scratchpad.get("researcher_a_answer", "(no answer)")
    b_ans = state.scratchpad.get("researcher_b_answer", "(no answer)")
    brief = (
        f"[Researcher A answered]: {a_ans[:800]}\n\n"
        f"[Researcher B answered]: {b_ans[:800]}"
    )
    state = append_message(state, AgentMessage(role="user", content=brief))
    return await writer.step(state)


# ── Researcher wrappers: stash their final answer in scratchpad ────────────────

async def research_a_step(state: GraphState) -> GraphState:
    state = await researcher_a.step(state)
    answer = state.messages[-1].content if state.messages else ""
    return set_scratchpad(state, "researcher_a_answer", answer)


async def research_b_step(state: GraphState) -> GraphState:
    state = await researcher_b.step(state)
    answer = state.messages[-1].content if state.messages else ""
    return set_scratchpad(state, "researcher_b_answer", answer)


# ── Graph ─────────────────────────────────────────────────────────────────────

graph = (
    Graph("research_team")
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
    state = GraphState(goal="How is AI being used in climate science, and what are the main risks?")
    final = asyncio.run(graph.run(state))
    print(final.messages[-1].content)
