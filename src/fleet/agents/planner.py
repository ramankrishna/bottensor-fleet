from __future__ import annotations

from fleet.core.messages import AgentMessage
from fleet.core.state import GraphState, append_message, set_scratchpad

_REACT_SYSTEM = """\
You are a goal-decomposition planner. Given a high-level objective, produce a
numbered list of concrete, actionable sub-goals. Think step by step (ReAct style):
Thought → Action → Observation. Return only the final numbered list."""

_USER_TEMPLATE = "Decompose this goal into sub-goals:\n\n{goal}"


class Planner:
    """Node that decomposes state.goal into sub-goals via ReAct prompting."""

    def __init__(self, llm: object) -> None:
        self._llm = llm

    async def __call__(self, state: GraphState) -> GraphState:
        return await self.plan(state)

    async def plan(self, state: GraphState) -> GraphState:
        llm = state.metadata.get("llm") or self._llm
        if llm is None:
            subgoals = f"(planner not configured — goal: {state.goal})"
        else:
            msgs = [
                AgentMessage(role="system", content=_REACT_SYSTEM),
                AgentMessage(role="user", content=_USER_TEMPLATE.format(goal=state.goal)),
            ]
            reply = await llm.complete(msgs)
            subgoals = reply.content or ""

        state = set_scratchpad(state, "planner", {"subgoals": subgoals})
        state = append_message(
            state,
            AgentMessage(role="assistant", content=subgoals, agent="planner"),
        )
        return state
