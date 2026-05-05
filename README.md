# bottensor-fleet

> **Graph-native multi-agent orchestration for Python. BYO-key. Local-first. Ships with a UI.**

[![PyPI](https://img.shields.io/pypi/v/bottensor-fleet)](https://pypi.org/project/bottensor-fleet/)
[![Python](https://img.shields.io/pypi/pyversions/bottensor-fleet)](https://pypi.org/project/bottensor-fleet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Demo

![bottensor-fleet demo](docs/demo.gif)

---

## Install

```bash
pip install bottensor-fleet
```

---

## 30-second quickstart

```python
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

state = GraphState(goal="What's new in AI safety research this week?")
final = asyncio.run(g.run(state))
print(final.messages[-1].content)
```

```bash
ANTHROPIC_API_KEY=sk-ant-… python examples/solo_agent.py
```

---

## UI

```bash
fleet ui
```

Opens `http://localhost:5173`. Live graph visualization, agent log, and run
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

| Provider | Model prefix | Key env var |
|---|---|---|
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/` | `OPENAI_API_KEY` |
| Google Gemini | `google/` | `GEMINI_API_KEY` |
| Groq | `groq/` | `GROQ_API_KEY` |
| Mistral | `mistral/` | `MISTRAL_API_KEY` |
| Ollama (local) | `ollama/` | _(none needed)_ |

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
    .add_edge("fixer",    "reviewer")  # loop
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

| Tool | Description |
|---|---|
| `web_search` | DuckDuckGo text search |
| `web_fetch` | Fetch a URL, return plain text |
| `python_exec` | Execute Python in a subprocess |
| `read_file` | Read from `FLEET_WORKSPACE` |
| `write_file` | Write to `FLEET_WORKSPACE` |
| `list_files` | List files under `FLEET_WORKSPACE` |

Custom tools: decorate any `async def` with `@tool`.

---

## CLI

```
fleet new <name>          scaffold a new graph project
fleet run <module> <goal> run a graph from the command line
fleet ui                  start the live dashboard
fleet add-agent <name>    scaffold an agent module
fleet ls                  list all recorded runs
fleet replay <run-id>     replay a finished run's events
```

---

## ⚠️ Security note

`python_exec` runs code in an **unsandboxed subprocess** in v0.1.
Only use it with trusted agents and goals.  A Docker sandbox is on the v0.2
roadmap.

---

## Roadmap

| Version | Focus |
|---|---|
| **v0.1** (now) | Core runtime, ReAct agents, FastAPI server, React UI |
| **v0.2** | Docker sandbox for `python_exec`, streaming token output |
| **v0.3** | Distributed scheduler (Redis task queue), multi-process workers |
| **v0.4** | Persistent vector memory (ChromaDB / pgvector), skill marketplace |

---

## License

MIT © 2026 Bottensor.  See [LICENSE](LICENSE).

Built with [polyrt](https://github.com/bottensor/polyrt) ·
[ReactFlow](https://reactflow.dev) · [Zustand](https://zustand-demo.pmnd.rs) ·
[Tailwind CSS](https://tailwindcss.com) · [FastAPI](https://fastapi.tiangolo.com)
