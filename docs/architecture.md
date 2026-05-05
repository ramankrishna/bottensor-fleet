# Architecture

## Overview

```
User / CLI / REST API
        │
        ▼
  ┌─────────────┐
  │   Graph      │  Fluent builder.  Validates topology (BFS orphan check).
  │  (builder)   │  Produces a CompiledGraph.
  └──────┬──────┘
         │ .compile()
         ▼
  ┌─────────────┐
  │  Compiled   │
  │   Graph     │  Holds the node/edge adjacency list + entry/exit.
  └──────┬──────┘
         │ .run(state)
         ▼
  ┌─────────────┐        ┌─────────────────┐
  │  Scheduler  │◄──────►│   EventBus      │  Pub/sub: emit(event) → callbacks
  │             │        │  (asyncio queue)│
  └──────┬──────┘        └────────┬────────┘
         │                        │
         │ per-step checkpoint     │ per-event emit
         ▼                        ▼
  ┌─────────────┐        ┌─────────────────┐
  │ Checkpoint  │        │  WebSocket      │
  │  Backend    │        │  Handler        │  Normalises events → JSON frames
  │ (SQLite /   │        │  (per run)      │  Heartbeat ping every 30 s
  │  Redis)     │        └────────┬────────┘
  └─────────────┘                 │
                                  ▼
                         ┌─────────────────┐
                         │   React UI      │  useFleetWS hook reconnects
                         │  (Vite + Zustand│  with exponential back-off.
                         │   + ReactFlow)  │  Zustand store drives render.
                         └─────────────────┘
```

---

## Component breakdown

### `Graph` and `CompiledGraph` (`src/fleet/core/graph.py`)

`Graph` is a fluent builder that accumulates `Node` and `Edge` dataclasses.
`compile()` runs a BFS from the entry node to detect orphan nodes, then
returns a `CompiledGraph`.

`CompiledGraph.run(state)` lazily imports `Scheduler` and `EventBus` to
avoid the circular import that would arise if `graph.py` imported
`scheduler.py` at module load time.

### `GraphState` (`src/fleet/core/state.py`)

Immutable Pydantic v2 model.  All state transitions go through three pure
reducers:

- `append_message(state, msg)` — adds to the message list
- `set_scratchpad(state, node, value)` — sets a per-node scratchpad slot
- `merge_metadata(state, data)` — shallow-merges metadata

The scheduler never mutates state in place; it always calls `model_copy`.

### `Scheduler` (`src/fleet/core/scheduler.py`)

Drives a BFS/topological walk:

1. Determine which nodes are ready (all predecessors finished).
2. Fan-out: `asyncio.gather(*[_run_node(n, state) for n in ready])`.
3. Merge: collect new messages / scratchpad writes from all parallel branches.
4. Checkpoint: persist merged state.
5. Repeat until the exit node finishes or `max_steps` is reached.

**Fan-out merge** — the base message count before fan-out is recorded;
after gather, only messages appended *after* that offset are collected from
each branch, then concatenated onto the base state.  Scratchpad and metadata
are shallow-merged (last-write-wins per key).

### `EventBus` (`src/fleet/core/scheduler.py`)

Thin pub/sub: `subscribe(callback)` registers an `async def callback(event)`
that is called for every `emit(event_dict)`.  The WebSocket handler
subscribes one callback per connected client.

### `Agent` (`src/fleet/agents/agent.py`)

Implements a ReAct loop:

```
for _ in range(max_iters):
    response = await llm.complete(messages, tools=schemas)
    state = append_message(state, response)
    if not response.tool_calls:
        break
    for tc in response.tool_calls:
        result = await _dispatch(tc.id, tc.name, tc.arguments)
        state = append_message(state, result)
```

Stops when the LLM emits no tool calls, or when `max_iters` is reached.

### `FleetLLM` (`src/fleet/providers/client.py`)

Wraps [polyrt](https://github.com/bottensor/polyrt) — a thin, backend-agnostic
LLM client.  Three retries with exponential back-off via `tenacity` on
`polyrt.BackendError`.

Converts `AgentMessage` ↔ `polyrt.Message` and maps `polyrt.ToolCall` back
to `fleet.core.messages.ToolCall`.

### FastAPI server (`src/fleet/server/`)

- `POST /api/runs` — creates a `RunRecord`, starts `_execute_run` as a
  background `asyncio.Task`.
- `GET  /api/runs/{id}` — returns run status.
- `POST /api/runs/{id}/pause|resume|kill` — lifecycle control.
- `GET  /api/graphs` — lists importable graph modules.
- `GET  /api/providers` — lists provider names from the polyrt registry.
- `WS   /ws/runs/{id}` — subscribes to the run's `EventBus` and streams
  JSON frames; sends a `ping` frame every 30 s.

### React UI (`ui/`)

| Layer | Technology |
|---|---|
| Build | Vite 8 + `@tailwindcss/vite` (Tailwind v4) |
| State | Zustand 5 |
| Graph | ReactFlow 11 |
| Fonts | Outfit · Instrument Serif · DM Mono (Google Fonts) |
| WS | `useFleetWS` — exponential back-off reconnect (500 ms → 30 s) |

---

## Data flow for a single run

```
POST /api/runs
  → RunRecord created, task spawned
  → CompiledGraph.run(state) called in background
      → Scheduler fans out nodes
          → each Agent.step() calls FleetLLM → polyrt → provider API
          → tool calls dispatched, results appended
          → EventBus.emit(node_started / tool_called / …)
              → WS handler queues JSON frame
              → client useFleetWS receives frame
              → dispatchEvent(frame) → Zustand store update
              → React re-renders GraphCanvas + AgentPanel
      → Scheduler checkpoints merged state
      → EventBus.emit(run_finished)
          → WS handler sends final frame, client stops reconnecting
```

---

## Key design decisions

**Immutable state** — pure reducers make checkpointing trivial: serialize `GraphState` to JSON, deserialize on resume.  No hidden mutation to track.

**Lazy scheduler import** — breaks the `graph → scheduler → graph` circular import without restructuring packages.

**Tailwind v4 CSS-first config** — the `@theme {}` block in `tokens.css` is the single source of truth; `tailwind.config.js` is only for IDE IntelliSense.

**polyrt** — a thin abstraction over provider SDKs.  Backends are registered at runtime so the core library has zero provider dependencies.
