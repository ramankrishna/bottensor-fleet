"""Single-agent graph: one researcher that answers with citations."""
from fleet import Graph, Agent
from fleet.memory import ReasoningBank
from fleet.providers.client import FleetLLM

_judge = FleetLLM("anthropic", "claude-sonnet-4-6")
_inducer = FleetLLM("anthropic", "claude-sonnet-4-6")

bank = ReasoningBank(
    scope="solo_researcher",
    judge_llm=_judge,
    induction_llm=_inducer,
)

researcher = Agent(
    name="researcher",
    goal="Answer the user's question with citations.",
    model="anthropic/claude-sonnet-4-6",
    tools=["web_search", "web_fetch"],
    memory_bank=bank,
)

graph = (
    Graph("solo")
    .add_node("researcher", researcher.step)
    .set_entry("researcher")
    .set_exit("researcher")
    .compile()
)

if __name__ == "__main__":
    import asyncio
    from fleet.core.state import GraphState

    state = GraphState(goal="What's new in interpretability research from Anthropic in 2026?")
    final = asyncio.run(graph.run(state))
    print(final.messages[-1].content)
