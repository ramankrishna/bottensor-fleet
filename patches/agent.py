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
        memory_bank: Any = None,
        memory_k: int = 5,
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
        self.memory_bank = memory_bank
        self.memory_k = memory_k

    async def step(self, state: GraphState) -> GraphState:
        if not state.messages and state.goal:
            if self.memory_bank is not None and self.memory_k > 0:
                memories = await self.memory_bank.retrieve(state.goal, k=self.memory_k)
                if memories:
                    state = append_message(
                        state,
                        AgentMessage(
                            role="system",
                            content=_format_memory_block(memories),
                        ),
                    )
            state = append_message(state, AgentMessage(role="user", content=state.goal))
        elif state.messages and state.messages[-1].role != "user":
            # Ensure conversation ends with a user message so the LLM can respond.
            # When a prior node left an assistant message at the tail, inject this
            # agent's goal (or the graph-level goal) as a handoff prompt.
            handoff = self.goal or state.goal or "Continue."
            # Resolve scratchpad references in the goal: replace scratchpad['key']
            # with actual values so each agent gets its specific task
            import re
            resolved = handoff
            for match in re.finditer(r"scratchpad\[(['\"])(\w+)\1\]", handoff):
                key = match.group(2)
                val = state.scratchpad.get(key, "")
                if val:
                    resolved = resolved.replace(match.group(0), str(val))
            prompt = f"[{self.name}] {resolved}"
            state = append_message(state, AgentMessage(role="user", content=prompt))

        tool_schemas = [to_anthropic(t) for t in self.tools if get_tool(t) is not None]

        # Get event_bus from the scheduler registry if available
        from fleet.core.scheduler import _EVENT_BUS_REGISTRY
        _run_id = state.metadata.get("run_id", "")
        _bus = _EVENT_BUS_REGISTRY.get(_run_id)
        _node = state.metadata.get("_current_node", self.name)

        async def _emit(event: dict) -> None:
            if _bus is not None:
                try:
                    await _bus.emit(event)
                except Exception:
                    pass

        for _ in range(self.max_iters):
            response = await self.llm.complete(
                state.messages,
                tools=tool_schemas or None,
            )
            state = append_message(state, response)

            # Emit state_update with the LLM response text
            if response.content:
                await _emit({
                    "type": "state_update",
                    "node": _node,
                    "content": response.content[:500],
                })

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                await _emit({
                    "type": "tool_call",
                    "node": _node,
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                })
                result_msg = await self._dispatch(tc.id, tc.name, tc.arguments)
                state = append_message(state, result_msg)
                result_content = ""
                if result_msg.tool_results:
                    result_content = result_msg.tool_results[0].content
                await _emit({
                    "type": "tool_result",
                    "node": _node,
                    "tool_name": tc.name,
                    "result": result_content[:500] if result_content else "",
                })

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


def _format_memory_block(memories: list[Any]) -> str:
    lines = ["Relevant past learnings (from prior trajectories on similar tasks):", ""]
    for i, m in enumerate(memories, start=1):
        lines.append(f"{i}. {m.title}")
        lines.append(f"   {m.description}")
        lines.append(f"   {m.content}")
        lines.append("")
    return "\n".join(lines).rstrip()
