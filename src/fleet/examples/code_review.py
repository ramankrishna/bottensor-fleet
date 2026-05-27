"""planner → reviewer → fixer ⟲ reviewer (conditional cycle).

reviewer sets scratchpad['approved'] = True/False.
If False the graph loops back to fixer (up to MAX_CYCLES times).
If True it exits.

Demonstrates: cycles, conditional edges, scratchpad-driven routing.
"""
import asyncio

from fleet import Agent, Graph
from fleet.core.state import GraphState, append_message, set_scratchpad
from fleet.agents.agent import AgentMessage

MAX_CYCLES = 3

# ── Agents ────────────────────────────────────────────────────────────────────

planner = Agent(
    name="planner",
    goal=(
        "Read the user's goal and output a short Python function that fulfils it. "
        "Return ONLY the function code, no prose."
    ),
    model="anthropic/claude-sonnet-4-6",
    tools=[],
    max_iters=1,
)

reviewer = Agent(
    name="reviewer",
    goal=(
        "Review the most recent Python code in the conversation. "
        "If it is correct and has no bugs, reply ONLY with the word APPROVED. "
        "Otherwise reply ONLY with a JSON object: "
        '{"approved": false, "feedback": "<one-sentence description of the issue>"}'
    ),
    model="anthropic/claude-sonnet-4-6",
    tools=[],
    max_iters=1,
)

fixer = Agent(
    name="fixer",
    goal=(
        "Fix the Python code based on the reviewer's feedback stored in "
        "scratchpad['review_feedback']. Return ONLY the corrected function code."
    ),
    model="anthropic/claude-sonnet-4-6",
    tools=[],
    max_iters=1,
)


# ── Node wrappers ─────────────────────────────────────────────────────────────

async def plan_step(state: GraphState) -> GraphState:
    return await planner.step(state)


async def review_step(state: GraphState) -> GraphState:
    import json

    state = await reviewer.step(state)
    verdict = (state.messages[-1].content or "").strip()
    if verdict.upper() == "APPROVED":
        state = set_scratchpad(state, "approved", True)
    else:
        try:
            parsed = json.loads(verdict)
            approved = bool(parsed.get("approved", False))
            feedback = parsed.get("feedback", verdict)
        except ValueError:
            approved = False
            feedback = verdict
        state = set_scratchpad(state, "approved", approved)
        state = set_scratchpad(state, "review_feedback", feedback)
        # Track cycle depth to avoid infinite loops when tests mock the LLM.
        cycles = int(state.scratchpad.get("cycles", 0)) + 1
        state = set_scratchpad(state, "cycles", cycles)
    return state


async def fix_step(state: GraphState) -> GraphState:
    feedback = state.scratchpad.get("review_feedback", "")
    if feedback:
        state = append_message(
            state,
            AgentMessage(role="user", content=f"Reviewer feedback: {feedback}"),
        )
    return await fixer.step(state)


# ── Routing helpers ───────────────────────────────────────────────────────────

def _approved(s: GraphState) -> bool:
    return bool(s.scratchpad.get("approved", False))

def _needs_fix(s: GraphState) -> bool:
    cycles = int(s.scratchpad.get("cycles", 0))
    return not _approved(s) and cycles < MAX_CYCLES


# ── Graph ─────────────────────────────────────────────────────────────────────

graph = (
    Graph("code_review")
    .add_node("planner",  plan_step)
    .add_node("reviewer", review_step)
    .add_node("fixer",    fix_step)
    # Linear edges
    .add_edge("planner",  "reviewer")
    # Conditional: reviewer → fixer if not approved, else exit
    .add_edge("reviewer", "fixer",    cond=_needs_fix)
    # Loop back: fixer → reviewer
    .add_edge("fixer",    "reviewer")
    .set_entry("planner")
    .set_exit("reviewer")
    .compile()
)

if __name__ == "__main__":
    state = GraphState(goal="Write a Python function that returns the nth Fibonacci number.")
    final = asyncio.run(graph.run(state))
    approved = final.scratchpad.get("approved", False)
    cycles   = final.scratchpad.get("cycles", 0)
    print(f"Approved: {approved}  |  Review cycles: {cycles}")
    print(final.messages[-1].content)
