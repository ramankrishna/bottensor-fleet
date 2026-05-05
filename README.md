# bottensor-fleet

> **Graph-native multi-agent orchestration for Python. BYO-key. Local-first. Ships with a UI.**

[![PyPI](https://img.shields.io/pypi/v/bottensor-fleet)](https://pypi.org/project/bottensor-fleet/)
[![Python](https://img.shields.io/pypi/pyversions/bottensor-fleet)](https://pypi.org/project/bottensor-fleet/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> **v0.1.1** — public release | [PyPI](https://pypi.org/project/bottensor-fleet/) | Apache-2.0

---

## Demo

![bottensor-fleet demo](docs/demo.gif)

---

## Install

```bash
# Recommended — includes web_search and web_fetch tools
pip install 'bottensor-fleet[search]'

# Minimal core (no web tools)
pip install bottensor-fleet

# Everything (web tools + Redis checkpoint)
pip install 'bottensor-fleet[all]'
```

### Extras

| Extra | Adds | Install |
|---|---|---|
| `[search]` | `web_search`, `web_fetch` tools (DuckDuckGo) | `pip install 'bottensor-fleet[search]'` |
| `[redis]` | `RedisCheckpoint` backend | `pip install 'bottensor-fleet[redis]'` |
| `[all]` | Both of the above | `pip install 'bottensor-fleet[all]'` |

---

## 30-second quickstart

```bash
pip install 'bottensor-fleet[search]'
fleet new my_agent
```

That generates a ready-to-run `my_agent.py`:

```python
from fleet import Agent, Graph
from fleet.core.state import GraphState
import asyncio

agent = Agent(
    name="agent",
    goal="Complete the user goal.",
    model="anthropic/claude-sonnet-4-6",  # or openai/gpt-4o, ollama/llama3, …
    tools=["web_search"],
)

graph = (
    Graph("my_agent")
    .add_node("agent", agent.step)
    .set_entry("agent")
    .set_exit("agent")
    .compile()
)

if __name__ == "__main__":
    state = GraphState(goal="What's new in AI safety research this week?")
    final = asyncio.run(graph.run(state))
    print(final.messages[-1].content)
```

```bash
ANTHROPIC_API_KEY=sk-ant-… python my_agent.py
```

---

## UI

```bash
fleet ui
```

Opens `http://localhost:8765`. Live graph visualization, agent log, and run
controls — all wired to the local WebSocket server.

![bottensor-fleet UI screenshot](docs/ui-screenshot.png)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Graph (fluent builder)                                 │
│    .add_node()  .add_edge(cond=...)  .compile()         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Scheduler                                              │
│    BFS walk · asyncio.gather fan-out · state merge      │
│    EventBus pub/sub · per-step checkpoint               │
└──────┬───────────────────────────────┬──────────────────┘
       │                               │
       ▼                               ▼
┌────────────┐               ┌─────────────────────┐
│ Checkpoint │               │  FastAPI + WS       │
│ SQLite /   │               │  /api/runs CRUD     │
│ Redis      │               │  /ws/runs/{id}      │
└────────────┘               └──────────┬──────────┘
                                        │
                                        ▼
                             ┌─────────────────────┐
                             │  React UI           │
                             │  Zustand · ReactFlow│
                             │  Tailwind v4        │
                             └─────────────────────┘
```

Each node in the graph is a plain `async def (GraphState) -> GraphState`.
Bring your own agents, or use the built-in `Agent` class for a full ReAct loop.

---

## Provider support

| Provider | Model string | Key env var |
|---|---|---|
| Anthropic | `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Ollama (local) | `ollama/llama3` | _(none needed)_ |
| MLX (Apple Silicon) | `mlx/<model>` | _(none needed)_ |

Backends are powered by [polyrt](https://github.com/bottensor/polyrt).
Mix providers freely — each `Agent` picks its own `model`.

---

## Multi-agent patterns

### Parallel fan-out

```python
g = (
    Graph("team")
    .add_node("planner",  planner.step)
    .add_node("worker_a", worker_a.step)
    .add_node("worker_b", worker_b.step)
    .add_node("writer",   writer.step)
    .add_edge("planner",  "worker_a")
    .add_edge("planner",  "worker_b")   # fan-out
    .add_edge("worker_a", "writer")
    .add_edge("worker_b", "writer")     # merge
    .set_entry("planner")
    .set_exit("writer")
    .compile()
)
```

`worker_a` and `worker_b` execute concurrently via `asyncio.gather`.

### Conditional cycles

```python
def _needs_fix(s): return not s.scratchpad.get("approved")

g = (
    Graph("review")
    .add_node("reviewer", review_step)
    .add_node("fixer",    fix_step)
    .add_edge("reviewer", "fixer",    cond=_needs_fix)
    .add_edge("fixer",    "reviewer")  # loop back
    .set_entry("reviewer")
    .set_exit("reviewer")
    .compile()
)
```

---

## How it compares

| Feature | **bottensor-fleet** | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Graph-native | ✅ | ✅ | ❌ role-based | ❌ conversation-first |
| No framework lock-in | ✅ plain Python | ⚠️ LangChain | ⚠️ CrewAI patterns | ⚠️ AutoGen patterns |
| Multi-provider | ✅ polyrt | ✅ | ✅ | ✅ |
| Built-in UI | ✅ React dashboard | ❌ | ❌ | ❌ |
| Checkpointing | ✅ SQLite / Redis | ✅ | ❌ | ❌ |
| Parallel fan-out | ✅ asyncio.gather | ✅ | ❌ | ⚠️ partial |
| Conditional cycles | ✅ edge `cond=` | ✅ | ❌ | ❌ |

---

## Built-in tools

| Tool | Extra needed | Description |
|---|---|---|
| `web_search` | `[search]` | DuckDuckGo text search, returns JSON |
| `web_fetch` | `[search]` | Fetch a URL, return plain text |
| `python_exec` | _(core)_ | Execute Python in a subprocess ⚠️ |
| `read_file` | _(core)_ | Read from `FLEET_WORKSPACE` |
| `write_file` | _(core)_ | Write to `FLEET_WORKSPACE` |
| `list_files` | _(core)_ | List files under `FLEET_WORKSPACE` |

Custom tools: decorate any `async def` with `@tool`.

---

## CLI

```
fleet --version               show installed version
fleet new <name>              scaffold a new graph (runnable Agent + Graph)
fleet run <module> --goal … run a graph from the command line
fleet ui                      start the live dashboard (http://localhost:8765)
fleet add-agent <file> …     append an Agent node to an existing graph
fleet ls                      list runs (run_id · saved_at · goal)
fleet replay <run-id>         re-run from a saved checkpoint
```

---

## ⚠️ Security note

`python_exec` runs code in an **unsandboxed subprocess** in v0.1.x.
Only use it with trusted agents and goals. A Docker sandbox is on the v0.2
roadmap.

---

## Roadmap

| Version | Focus |
|---|---|
| **v0.1.1** (now) | Core runtime, ReAct agents, FastAPI server, React UI, optional extras |
| **v0.2** | Docker sandbox for `python_exec`, streaming token output |
| **v0.3** | Distributed scheduler (Redis task queue), multi-process workers |
| **v0.4** | Persistent vector memory (ChromaDB / pgvector), skill marketplace |

---

## License

Apache-2.0 © 2026 Bottensor. See [LICENSE](LICENSE).

Built with [polyrt](https://github.com/bottensor/polyrt) ·
[ReactFlow](https://reactflow.dev) · [Zustand](https://zustand-demo.pmnd.rs) ·
[Tailwind CSS](https://tailwindcss.com) · [FastAPI](https://fastapi.tiangolo.com)
