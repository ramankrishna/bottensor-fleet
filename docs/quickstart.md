# Quickstart

## Install

```bash
pip install bottensor-fleet
```

Requires Python 3.11+. For the development environment:

```bash
git clone https://github.com/ramankrishna/bottensor-fleet
cd bottensor-fleet
uv sync --extra dev
```

---

## Your first graph — solo agent

```python
# examples/solo_agent.py
from fleet import Graph, Agent

researcher = Agent(
    name="researcher",
    goal="Answer the user's question with citations.",
    model="anthropic/claude-sonnet-4-6",
    tools=["web_search", "web_fetch"],
)

g = (
    Graph("solo")
    .add_node("researcher", researcher.step)
    .set_entry("researcher")
    .set_exit("researcher")
    .compile()
)

import asyncio
from fleet.core.state import GraphState

state = GraphState(goal="What's new in interpretability research from Anthropic in 2026?")
final = asyncio.run(g.run(state))
print(final.messages[-1].content)
```

Run it:

```bash
ANTHROPIC_API_KEY=sk-ant-… python examples/solo_agent.py
```

> **Tip:** fleet uses [polyrt](https://github.com/bottensor/polyrt) under the hood.
> Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — whichever backend you pass in
> the `model` field.

---

## Multi-agent parallel fan-out

```python
# examples/research_team.py  (abbreviated)
g = (
    Graph("research_team")
    .add_node("planner",      plan_step)
    .add_node("researcher_a", research_a_step)
    .add_node("researcher_b", research_b_step)
    .add_node("writer",       write_step)
    .add_edge("planner",      "researcher_a")
    .add_edge("planner",      "researcher_b")   # fan-out
    .add_edge("researcher_a", "writer")
    .add_edge("researcher_b", "writer")         # merge
    .set_entry("planner")
    .set_exit("writer")
    .compile()
)
```

`researcher_a` and `researcher_b` execute concurrently via `asyncio.gather`.
Their scratchpad writes are merged before the writer node runs.

---

## Conditional cycles

```python
# examples/code_review.py  (abbreviated)
g = (
    Graph("code_review")
    .add_node("planner",  plan_step)
    .add_node("reviewer", review_step)
    .add_node("fixer",    fix_step)
    .add_edge("planner",  "reviewer")
    .add_edge("reviewer", "fixer",   cond=lambda s: not s.scratchpad.get("approved"))
    .add_edge("fixer",    "reviewer")           # loop back
    .set_entry("planner")
    .set_exit("reviewer")
    .compile()
)
```

`cond` is a `GraphState → bool` callable. When it returns `False` the edge is
skipped. When all outgoing edges from a node are skipped the scheduler treats
that node as the terminal node (equivalent to reaching the exit).

---

## Launching the UI

```bash
fleet ui
```

Opens `http://localhost:8765` with:

- **Left rail** — Run controls (goal, graph selector, Start / Pause / Resume / Kill)
- **Center** — ReactFlow graph; nodes show running / waiting state
- **Right rail** — Message / tool log for the selected agent

Provider keys are read from the environment of the `fleet ui` process — the
UI does not accept API keys through forms.

Start a run from the UI, or use the REST API directly:

```bash
curl -X POST http://localhost:8765/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Summarise today'\''s AI news", "graph_module": "examples.solo_agent", "backend": "anthropic"}'
```

---

## CLI reference

```
fleet new <name>          scaffold a new graph project
fleet run <module> <goal> run a compiled graph from the CLI
fleet ui                  start the dashboard
fleet add-agent <name>    scaffold an agent module
fleet ls                  list all recorded runs
fleet replay <run-id>     replay events from a finished run
```

---

## Built-in tools

| Name | Description |
|---|---|
| `web_search` | DuckDuckGo text search, returns JSON |
| `web_fetch` | Fetch a URL, returns plain text |
| `python_exec` | Execute Python in a subprocess ⚠️ unsandboxed in v0.1 |
| `read_file` | Read a file from `FLEET_WORKSPACE` |
| `write_file` | Write a file to `FLEET_WORKSPACE` |
| `list_dir` | List entries under a path inside `FLEET_WORKSPACE` |

Pass tool names as strings to `Agent(tools=[...])`. All tools are registered
via the `@tool` decorator and discoverable with `fleet.tools.get_tool(name)`.

---

## Custom tools

```python
from fleet import tool

@tool
async def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())

# Use by name in any Agent:
agent = Agent(name="counter", tools=["count_words"], ...)
```

---

## Custom skills (reusable graph-state transforms)

A skill is an async `GraphState -> GraphState` function registered by name.
Drop one in your project and reference it from a node.

```python
from fleet.core.state import GraphState, append_message
from fleet.core.messages import AgentMessage
from fleet.skills import skill, get_skill

@skill
async def echo_goal(state: GraphState) -> GraphState:
    """Append the current goal back as an assistant message."""
    return append_message(state, AgentMessage(role="assistant", content=state.goal))

entry = get_skill("echo_goal")
```

---

## Checkpointing

```python
from fleet.core.checkpoint import SQLiteCheckpoint

g = (
    Graph("solo")
    .add_node(...)
    .compile(backend=SQLiteCheckpoint())   # persists to ~/.fleet/checkpoints.db
)
```

For Redis:

```python
from fleet.core.checkpoint import RedisCheckpoint
g = Graph(...).compile(backend=RedisCheckpoint("redis://localhost:6379"))
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `FLEET_WORKSPACE` | `~/.fleet/workspace` | Root for file tool operations |
| `ANTHROPIC_API_KEY` | — | Anthropic backend key |
| `OPENAI_API_KEY` | — | OpenAI backend key |

Export the relevant key(s) **before** launching `fleet ui` — the UI reads
credentials from the server process's environment and does not accept keys
through forms.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
fleet ui
```
