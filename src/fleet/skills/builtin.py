from __future__ import annotations

from fleet.core.messages import AgentMessage
from fleet.core.state import GraphState, append_message, set_scratchpad
from fleet.skills.registry import skill

_PLAN_PROMPT = (
    "You are a planning assistant. Break the following goal into 3-5 concrete, "
    "numbered sub-steps. Return only the numbered list.\n\nGoal: {goal}"
)

_SUMMARIZE_PROMPT = (
    "Summarize the conversation so far in 2-3 sentences. Be concise."
)

_CRITIQUE_PROMPT = (
    "Critique the last assistant response. Identify strengths and any weaknesses "
    "or factual gaps. Be specific."
)


@skill
async def plan(state: GraphState) -> GraphState:
    """Decompose the goal into numbered sub-steps; stored in scratchpad['plan']."""
    # Placeholder: real implementation calls FleetLLM injected via state.metadata.
    llm = state.metadata.get("llm")
    if llm is not None:
        prompt = _PLAN_PROMPT.format(goal=state.goal)
        msgs = [AgentMessage(role="user", content=prompt)]
        reply = await llm.complete(msgs)
        subgoals = reply.content or ""
    else:
        subgoals = f"(plan pending — goal: {state.goal})"

    state = set_scratchpad(state, "plan", {"subgoals": subgoals})
    state = append_message(state, AgentMessage(role="assistant", content=subgoals))
    return state


@skill
async def summarize(state: GraphState) -> GraphState:
    """Summarize the conversation; stored in scratchpad['summary']."""
    llm = state.metadata.get("llm")
    if llm is not None:
        ctx = "\n".join(
            f"{m.role}: {m.content}" for m in state.messages if m.content
        )
        msgs = [
            AgentMessage(role="user", content=f"{_SUMMARIZE_PROMPT}\n\n{ctx}"),
        ]
        reply = await llm.complete(msgs)
        summary = reply.content or ""
    else:
        summary = f"(summary pending — {len(state.messages)} messages)"

    state = set_scratchpad(state, "summary", {"text": summary})
    state = append_message(state, AgentMessage(role="assistant", content=summary))
    return state


@skill
async def critique(state: GraphState) -> GraphState:
    """Critique the last assistant message; stored in scratchpad['critique']."""
    last_assistant = next(
        (m for m in reversed(state.messages) if m.role == "assistant" and m.content),
        None,
    )
    llm = state.metadata.get("llm")
    if llm is not None and last_assistant is not None:
        msgs = [
            AgentMessage(
                role="user",
                content=f"{_CRITIQUE_PROMPT}\n\nResponse to critique:\n{last_assistant.content}",
            )
        ]
        reply = await llm.complete(msgs)
        feedback = reply.content or ""
    else:
        feedback = "(critique pending — no assistant message found)" if last_assistant is None else "(critique pending)"

    state = set_scratchpad(state, "critique", {"feedback": feedback})
    state = append_message(state, AgentMessage(role="assistant", content=feedback))
    return state
