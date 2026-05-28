# bottensor-fleet

> Graph-native multi-agent fleet for Python. BYO-key. Local-first. Ships with a UI.

[![PyPI](https://img.shields.io/pypi/v/bottensor-fleet.svg)](https://pypi.org/project/bottensor-fleet/)
[![Python](https://img.shields.io/pypi/pyversions/bottensor-fleet.svg)](https://pypi.org/project/bottensor-fleet/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

<!-- demo.gif placeholder — will be added after capture run -->

## Why

Most multi-agent frameworks are heavy and locked to one ecosystem. LangGraph is tied to LangChain. CrewAI is opinionated about roles. AutoGen is conversation-first. `bottensor-fleet` is a small, graph-native runtime that runs anywhere Python runs, lets you bring your own provider key, and ships with a real UI in the wheel.

## Install

```bash
pip install 'bottensor-fleet[search]'
export ANTHROPIC_API_KEY=sk-ant-...
```

Extras: `[search]` adds web tools, `[redis]` adds Redis checkpointing, `[memory]` adds ReasoningBank + MaTTS (SQLite-vec store + MiniLM embedder), `[all]` gets everything.

## 30-second example

```python
import asyncio
from fleet import Agent, Graph
from fleet.core.state import GraphState
from fleet.providers.client import FleetLLM

llm = FleetLLM("anthropic", "claude-sonnet-4-6")
researcher = Agent(name="researcher", llm=llm, tools=["web_search", "web_fetch"])

graph = (
    Graph("solo")
    .add_node("researcher", researcher.step)
    .set_entry("researcher")
    .set_exit("researcher")
    .compile()
)

state = asyncio.run(graph.run(GraphState(goal="What is ReasoningBank?")))
print(state.messages[-1].content)
```

## UI

Export your provider key(s) in the same shell, then launch:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # and/or OPENAI_API_KEY, GEMINI_API_KEY, ...
fleet ui
```

Opens a local dashboard at `http://localhost:8765` with a live DAG view, per-agent logs, token spend, and run history.

The UI never accepts API keys through forms — credentials are read from the environment of the `fleet ui` process. Restart `fleet ui` after changing an exported key.

## CLI

| Command | What it does |
|---|---|
| `fleet new <name>` | Scaffold a new graph |
| `fleet run <graph.py>` | Run a graph from a file |
| `fleet replay <run_id>` | Re-run a past graph from its saved source path |
| `fleet examples [name]` | List bundled examples or extract one to the current directory |
| `fleet ui` | Launch the local dashboard |
| `fleet add-agent` | Append an agent to an existing graph |
| `fleet ls` | List past runs |
| `fleet --version` | Print version |

## Design

- **Graph-native:** DAGs with conditional edges and bounded cycles, executed async with `asyncio.gather` for parallel fan-out.
- **BYO-key:** Provider abstraction via [polyrt](https://pypi.org/project/polyrt/). Anthropic and OpenAI in the default install; MLX, Ollama, and others via polyrt extras.
- **Checkpointed:** Every run persists to SQLite (default) or Redis (opt-in via `[redis]` extra).
- **Tools and skills:** `@tool` decorator auto-derives JSON schemas from type hints. `@skill` for higher-level capabilities. Web search and fetch built in via the `[search]` extra.
- **UI in the wheel:** No separate Node install for users. The React + Vite frontend is bundled into the published wheel.

## Memory & self-improving agents

Opt-in **ReasoningBank** ([Ouyang et al., ICLR 2026](https://arxiv.org/abs/2509.25140)) gives any agent a persistent, embedding-indexed memory of past trajectories. Retrieval prepends the top-k relevant memories before each run; an async writeback judges the trajectory and distills 1–3 generalizable lessons that integrate into the bank via merge / link / insert.

**MaTTS (Memory-Aware Test-Time Scaling)** runs the same task `k` times in parallel and contrast-distills higher-quality memories than any single rollout can produce — useful for bootstrapping a new scope or one-shot high-stakes tasks.

```bash
pip install 'bottensor-fleet[memory]'
```

```python
from fleet import Agent, ReasoningBank
from fleet.providers.client import FleetLLM

llm  = FleetLLM("anthropic", "claude-sonnet-4-6")
bank = ReasoningBank(judge_llm=llm, induction_llm=llm, scope="research")

researcher = Agent(name="researcher", model="anthropic/claude-sonnet-4-6",
                   tools=["web_search"], memory_bank=bank, memory_k=5)
```

Two ready-to-run examples:

```bash
fleet examples learning_research_team    # planner + 2 researchers + writer, shared bank
fleet examples matts_solo                # single agent with MaTTS k=3
```

The dashboard's **Memory** tab (browse / search / edit / export / import) is wired to `/api/memory*`. See [docs/memory.md](docs/memory.md) for the full guide.

## Comparison

| | bottensor-fleet | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| Graph topology | ✅ DAG + cycles | ✅ | ❌ role-based | ❌ conversation |
| Provider-agnostic | ✅ via polyrt | ⚠️ via LangChain | ⚠️ | ⚠️ |
| Ships with UI | ✅ | ❌ | ❌ | ⚠️ Studio (separate) |
| Pip-install size | ~150 KB wheel | heavy | medium | heavy |
| LangChain dependency | ❌ | ✅ required | ❌ | ❌ |

## Roadmap

- **v0.2** ✅ — ReasoningBank + parallel MaTTS shipped. See [docs/memory.md](docs/memory.md).
- **v0.3** — MLX embedder hardening, sequential MaTTS, distributed scheduler.
- **v0.4** — Alternate vector backends (Redis / Postgres), cloud deploy templates.

## Security

The `python_exec` tool is unsandboxed in v0.1.x. Do not run untrusted graphs. A Docker sandbox lands in v0.2.

## License

Apache-2.0. © 2026 Rama Krishna Bachu.

## Acknowledgements

Built on [polyrt](https://pypi.org/project/polyrt/). ReasoningBank design (v0.2) follows Ouyang et al., *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*, ICLR 2026.
