# Visual graph builder

The **Builder** tab in `fleet ui` is a drag-and-drop editor for multi-agent
graphs. You wire up agents on a canvas, run the graph live without leaving the
browser, and export the result as JSON or as a runnable Python file.

The builder produces a **GraphSpec** — a small JSON document the rest of fleet
treats as a first-class graph format. The same spec runs from the CLI
(`fleet run graph.json`) and from Python (`load_graph_spec(...)`).

## Launching the builder

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY, DEEPSEEK_API_KEY, …
fleet ui
```

Pick **builder** in the top nav. The dashboard never reads keys from the UI —
credentials come from the environment of the `fleet ui` process. Restart it
after exporting a new key.

## Authoring a graph

1. **Drag an Agent** from the left palette onto the canvas. The first node you
   drop is automatically marked as both entry and exit.
2. **Click the node** to configure it in the right side panel:
   - `name` / `provider` / `model` / `system prompt`
   - `tools` — multi-select over every `@tool` registered in this process
   - `memory_bank` — toggle ReasoningBank retrieval + writeback
   - `base_url` — shown only when the provider needs it (always for
     `custom`, optional for `deepseek`)
   - `entry` / `exit` pills — exactly one of each is required
3. **Connect nodes** by dragging from the bottom handle of one node to the top
   handle of another.
4. **Click an edge** to set a condition from the registry. JSON cannot
   reference arbitrary Python predicates — the only conditions that can fire
   are the ones registered server-side. Ship-built-ins: `always`,
   `max_steps_not_hit`, and the parametric `scratchpad_true:<key>`.
5. **Type a goal** in the toolbar, hit **Run**. Live status badges
   (`idle / running / waiting / done / error`) overlay the same canvas using
   the existing WebSocket event stream.
6. **Save / Open / Export .py** are in the toolbar:
   - *Save .json* downloads the GraphSpec.
   - *Open* loads a GraphSpec back into the canvas.
   - *Export .py* asks the server for a runnable Python rendering — useful
     when you want to drop down from the visual editor to code.

## GraphSpec format reference

```jsonc
{
  "version": "0.3",
  "name": "research_team",
  "nodes": [
    {
      "id": "planner",
      "type": "agent",
      "agent": {
        "name": "planner",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "system": "Break the goal into two subqueries.",
        "tools": [],
        "memory_bank": false,
        "base_url": null,
        "max_iters": 10
      },
      "position": {"x": 100, "y": 40}
    }
  ],
  "edges": [
    {"src": "planner", "dst": "researcher", "cond": null},
    {"src": "researcher", "dst": "writer", "cond": "scratchpad_true:has_findings"}
  ],
  "entry": "planner",
  "exit": "writer"
}
```

**Fields**

| Path | Required | Notes |
|---|---|---|
| `version` | yes | Must be `"0.3"` |
| `name` | yes | Free-form |
| `nodes[].id` | yes | Unique across the graph |
| `nodes[].type` | yes | Only `"agent"` is supported in v0.3 |
| `nodes[].agent.provider` | yes | One of `anthropic`, `claude`, `openai`, `deepseek`, `custom`, `ollama`, `gemini`, `mistral` |
| `nodes[].agent.model` | yes | Provider-specific model id |
| `nodes[].agent.system` | yes | System prompt (passed as the Agent's `goal`) |
| `nodes[].agent.tools` | no | Names registered via `@fleet.tool`. Unknown names are rejected at validation. |
| `nodes[].agent.memory_bank` | no | `true` constructs a default `ReasoningBank()` |
| `nodes[].agent.base_url` | conditional | Required when `provider == "custom"`. Optional default exists for `deepseek`. |
| `nodes[].position` | no | UI coords (the runtime ignores them; persisted so the builder can round-trip) |
| `edges[].cond` | no | A condition name registered server-side. `null` means unconditional. |
| `entry` / `exit` | yes | Must reference existing node ids |

## Providers

| Provider | Auth env var | Base URL |
|---|---|---|
| `anthropic` / `claude` | `ANTHROPIC_API_KEY` | hosted (Anthropic) |
| `openai` | `OPENAI_API_KEY` | hosted (OpenAI) |
| `deepseek` | `DEEPSEEK_API_KEY` (falls back to `OPENAI_API_KEY`) | defaults to `https://api.deepseek.com`; override with `base_url` |
| `custom` | `CUSTOM_API_KEY` (falls back to `OPENAI_API_KEY`) | required — points at any OpenAI-compatible endpoint (vLLM, Ollama, Together, LM Studio) |

## Running a spec from elsewhere

The same JSON runs from three places without modification:

```bash
# CLI
fleet run graph.json --goal "How is AI being used in climate science?"

# Or the visual builder's Run button (POST /api/runs/from-spec)

# Or Python
from fleet.graphspec import load_graph_spec
import asyncio
from fleet.core.state import GraphState

cg = load_graph_spec("graph.json")
asyncio.run(cg.run(GraphState(goal="…")))
```

## Trust model (read this before sharing specs)

A GraphSpec is a runnable program. It declares which providers to call, which
tools to invoke, and which model endpoints to hit. **Treat any spec you didn't
write yourself with the same caution you'd treat a Python script** — review it
before running.

Specifically:

- **Tools run in-process, unsandboxed.** `python_exec` executes arbitrary
  Python in this interpreter. The Builder shows a ⚠ warning when you add it
  to a node. Don't run a downloaded spec that includes `python_exec` unless
  you've read the surrounding system prompt and trust the author.
- **`memory_bank: true` writes to your local memory store.** A malicious spec
  could poison your bank with biased prior "lessons" that contaminate future
  unrelated runs.
- **`base_url` is restricted by an SSRF guard.** Non-http(s) schemes
  (`file://`, `gopher://`, …) and known cloud-metadata endpoints
  (`169.254.169.254`, `metadata.google.internal`, etc.) are rejected at spec
  validation. Localhost and RFC1918 ranges are allowed because local vLLM /
  Ollama is the whole point of `provider: "custom"`. Operators needing
  stronger network isolation should run fleet behind a sandbox or egress
  firewall — the validator catches the obvious shapes only.
- **Conditions are whitelist-only.** The JSON can only reference condition
  names registered with `fleet.graphspec.register_condition()`. There is no
  `eval` path. An unknown name is a validation error, not a silent fallback.

## Programmatic API

```python
from fleet.graphspec import (
    GraphSpec, NodeSpec, AgentSpec, EdgeSpec, Position,
    load_graph_spec, spec_to_python,
    register_condition, get_condition,
    SUPPORTED_PROVIDERS, validate_base_url,
)
```

- `load_graph_spec(spec | dict | json_str | path)` — returns a runnable
  `CompiledGraph`.
- `spec_to_python(spec)` — renders the spec as a runnable Python source file
  that mirrors the style of `fleet new`. Output passes `ruff check`.
- `register_condition(name, fn)` / `register_parametric_condition(prefix, factory)`
  — extend the condition registry from your own process before serving the UI.
- `validate_base_url(url)` — exposed for tests and for callers that want to
  apply the SSRF guard outside of spec validation.

## See also

- [`examples/research_team.json`](../examples/research_team.json) — reference
  spec that ships in the wheel.
- [`docs/memory.md`](memory.md) — when `memory_bank: true` actually does
  something interesting.
