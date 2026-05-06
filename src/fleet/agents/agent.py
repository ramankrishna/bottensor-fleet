from __future__ import annotations

from typing import Any, Protocol

from fleet.core.messages import AgentMessage, ToolResult
from fleet.core.state import GraphState, append_message
from fleet.tools.base import get_tool, to_anthropic


class LLMProtocol(Protocol):
    async def complete(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentMessage: ...

    @property
    def usage(self) -> Any: ...


class Agent:
    """ReAct agent: repeatedly calls the LLM and dispatches tool calls until
    the model returns a plain text response or *max_iters* is reached.

    Two construction styles are supported:

    Low-level (pass an LLM object directly):
        Agent(llm, tools=["web_search"], max_iters=10)

    High-level (pass model string, FleetLLM is constructed automatically):
        Agent(name="researcher", model="anthropic/claude-sonnet-4-6",
              goal="...", tools=["web_search"])
    """

    def __init__(
        self,
        llm: LLMProtocol | None = None,
        tools: list[str] | None = None,
        max_iters: int = 10,
        *,
        name: str = "agent",
        goal: str = "",
        model: str = "",
    ) -> None:
        if llm is None:
            if not model:
                raise ValueError("Provide either an llm object or a model string.")
            from fleet.providers.client import FleetLLM
            if "/" in model:
                backend_str, model_name = model.split("/", 1)
            else:
                backend_str, model_name = model, model
            llm = FleetLLM(backend_str, model_name)

        self.name = name
        self.goal = goal
        self.llm: LLMProtocol = llm
        self.tools: list[str] = tools or []
        self.max_iters = max_iters

    async def step(self, state: GraphState) -> GraphState:
        # Seed / re-seed so the conversation always ends with a user turn before
        # we hit the LLM. Three cases:
        #   1. empty messages: seed with the agent's own goal (preferred) or the graph goal
        #   2. last message is from a prior assistant: this is a multi-agent handoff,
        #      prompt the current agent to take over with a fresh user message
        #   3. otherwise (last was user or tool): leave as-is, already a valid turn
        prompt: str | None = None
        if not state.messages:
            prompt = self.goal or state.goal or None
        else:
            last = state.messages[-1]
            if last.role == "assistant":
                handoff = f"Now you ({self.name}) take over."
                if self.goal:
                    handoff += f" Your role: {self.goal}"
                if state.goal and state.goal not in (self.goal or ""):
                    handoff += f"\n\nOverall goal: {state.goal}"
                prompt = handoff
        if prompt:
            state = append_message(state, AgentMessage(role="user", content=prompt))

        tool_schemas = [to_anthropic(t) for t in self.tools if get_tool(t) is not None]
        if self.tools and not tool_schemas:
            import warnings
            unknown = [t for t in self.tools if get_tool(t) is None]
            warnings.warn(
                f"Agent {self.name!r} requested tools {unknown} but none are registered. "
                f"Did you forget `import fleet.tools.web` (or another tools module)?",
                RuntimeWarning,
                stacklevel=2,
            )

        for _ in range(self.max_iters):
            response = await self.llm.complete(
                state.messages,
                tools=tool_schemas or None,
            )
            state = append_message(state, response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                result_msg = await self._dispatch(tc.id, tc.name, tc.arguments)
                state = append_message(state, result_msg)

        return state

    async def _dispatch(
        self, call_id: str, name: str, arguments: dict[str, Any]
    ) -> AgentMessage:
        entry = get_tool(name)
        if entry is None:
            content = f"Error: tool '{name}' is not registered"
            is_error = True
        else:
            try:
                result = await entry["fn"](**arguments)
                content = str(result)
                is_error = False
            except Exception as exc:
                content = f"Error: {exc}"
                is_error = True

        return AgentMessage(
            role="tool",
            tool_results=[ToolResult(tool_call_id=call_id, content=content, is_error=is_error)],
        )
