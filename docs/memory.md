# Memory & Self-Evolving Agents

`bottensor-fleet` ships with **ReasoningBank**, an experience-augmented memory layer that lets agents learn from their own runs. It is an implementation of the design from Ouyang et al., [*ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*](https://arxiv.org/abs/2509.25140) (ICLR 2026), with one extension: **Memory-Aware Test-Time Scaling (MaTTS)** for higher-quality memory at the cost of extra compute.

The bank is **opt-in**. Existing graphs and agents run unchanged; you wire memory in only when you want it.

---

## Overview

A `ReasoningBank` is a small library that:

1. **Retrieves** the most relevant past learnings for a new task (top-k cosine search over embeddings) and injects them into the agent's context as a system message.
2. **Ingests** the trajectory after the run: an LLM judges success/failure, an inducer distills 1–3 generalizable memories, and a merge step decides whether each candidate is a brand-new memory, a refinement of an existing one, or a sibling that should be linked.

The result is an agent that gets measurably better on tasks similar to ones it has already attempted — without retraining, fine-tuning, or hand-curated prompt edits.

```
    ┌──────────┐  retrieve top-k     ┌────────────────────┐
    │   task   │ ──────────────────▶ │   ReasoningBank    │
    └──────────┘ ◀────── memories ── │  (store+embedder)  │
         │                           └─────────┬──────────┘
         ▼                                     ▲
    ┌──────────┐                               │  judge → induce → merge
    │  agent   │ ─── trajectory ───────────────┘
    └──────────┘
```

---

## Quick start

Install with the `[memory]` extra (pulls in `sqlite-vec` and `sentence-transformers`):

```bash
pip install 'bottensor-fleet[memory]'
```

Attach a bank to an agent:

```python
import asyncio
from fleet import Agent, Graph, ReasoningBank
from fleet.core.state import GraphState
from fleet.providers.client import FleetLLM

judge   = FleetLLM("anthropic", "claude-sonnet-4-6")
inducer = FleetLLM("anthropic", "claude-sonnet-4-6")

bank = ReasoningBank(
    scope="research",
    judge_llm=judge,
    induction_llm=inducer,
)

researcher = Agent(
    name="researcher",
    model="anthropic/claude-sonnet-4-6",
    tools=["web_search", "web_fetch"],
    goal="Answer the user's question with sources.",
    memory_bank=bank,
    memory_k=5,
)

graph = (
    Graph("solo")
    .add_node("researcher", researcher.step)
    .set_entry("researcher")
    .set_exit("researcher")
    .compile()
)

state = asyncio.run(graph.run(GraphState(goal="What is RAG?")))
print(state.messages[-1].content)
```

On the first run the bank is empty, so retrieval is a no-op; after the run the scheduler kicks off an async writeback that judges the trajectory and ingests 1–3 distilled memories. On the next run with a similar `goal`, those memories are retrieved and prepended as a system message.

Two ready-to-run examples ship in the wheel:

```bash
fleet examples learning_research_team    # planner + 2 researchers + writer, shared bank
fleet examples matts_solo                # single agent with MaTTS k=3
```

---

## `MemoryItem` schema

Every record in the bank is a `MemoryItem` (`fleet.memory.MemoryItem`):

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | Auto-generated `mem_<10 hex>`. |
| `title` | `str` | Short label, used in the UI. |
| `description` | `str` | One- or two-sentence summary of when the lesson applies. |
| `content` | `str` | The actual generalizable advice. |
| `source` | `"success" \| "failure" \| "matts_contrast" \| "manual"` | Where the item came from. |
| `task_signature` | `str` | Original task text. Lets you trace memories back to the task that produced them. |
| `scope` | `str` | Namespace (e.g., `"research_team"`, `"global"`). Retrieval and search are scope-filtered. |
| `embedding` | `list[float] \| None` | Set by the bank's embedder; required before `store.add()`. |
| `created_at` | `datetime` | UTC timestamp. |
| `last_used_at` | `datetime \| None` | Updated on retrieval (for future LRU pruning). |
| `use_count` | `int` | Incremented on retrieval. |
| `confidence` | `float` | 0..1; merged items inherit higher confidence. |
| `related_ids` | `list[str]` | Sibling links built by the `link` branch of the merge logic. |

The text the embedder sees is `f"{title}\n{description}\n{content}"` (`item.signature_text()`).

---

## Storage backends

The `MemoryStore` protocol (`fleet.memory.stores.MemoryStore`) is a tiny CRUD + vector-search interface. Two backends ship today:

### `SQLiteVecStore` (default)

Persistent, file-backed, uses the [`sqlite-vec`](https://github.com/asg017/sqlite-vec) extension for cosine similarity search.

- DB path: `$FLEET_MEMORY_DB`, else `~/.fleet/reasoning_bank.db`.
- Embedding dimension: fixed at construction (default 384, matches MiniLM).
- Schema: `memory_items` for the row data, `memory_vec` (vec0 virtual table) for the vectors, `memory_vec_map` to join rowid → memory id.

```python
from fleet.memory.stores.sqlite_vec import SQLiteVecStore
store = SQLiteVecStore(db_path="/tmp/bank.db", dim=384)
```

### `InMemoryStore` (tests, ephemeral runs)

Pure-Python dict + brute-force cosine. No dependencies, no persistence. The bank falls back to this automatically if `sqlite-vec` isn't installed.

```python
from fleet.memory.stores import MemoryStore
from fleet.memory.stores.inmemory import InMemoryStore
store: MemoryStore = InMemoryStore()
```

### Writing your own

Any object that satisfies the `MemoryStore` protocol works. The protocol is `@runtime_checkable`, so duck typing is fine.

---

## Embedders

The `Embedder` protocol is two methods: `dim: int` and `embed(texts) -> list[list[float]]`. v0.2 registers one embedder; MLX and polyrt are reserved for v0.3.

| Name | Class | Requires | Status |
|---|---|---|---|
| `minilm` | `MiniLMEmbedder` | `sentence-transformers` (default install via `[memory]`) — 384-d. | ✅ shipping |
| `mlx` | `MLXEmbedder` | `mlx-embeddings` (Apple Silicon). | 🧪 stub, lands in v0.3 |
| `polyrt` | `PolyRTEmbedder` | A polyrt backend that exposes an embeddings API. | 🧪 stub, lands in v0.3 |

```python
from fleet.memory import get_embedder
embedder = get_embedder("minilm")
```

Register a custom one:

```python
from fleet.memory import register_embedder

def factory(**kwargs):
    return MyEmbedder(**kwargs)

register_embedder("custom", factory)
```

MiniLM loads `sentence-transformers/all-MiniLM-L6-v2` lazily on first `embed()`, so importing `fleet.memory` is cheap.

---

## The induction pipeline

When a trajectory finishes, the bank runs three stages:

### 1. Judge

`fleet.memory.judge.judge_trajectory(trajectory, task, llm)` asks an LLM to decide `success` or `failure` and return a short rationale. The judge is strict — partial or ambiguous results are classified as failures. This is skipped if you call `ingest_trajectory(..., outcome="success")` directly.

### 2. Induction

`fleet.memory.induction.distill_memories(...)` renders the trajectory and asks an LLM to extract 1–3 generalizable `MemoryItem` records. Different prompts are used for success vs. failure (`prompts/induction_success.txt`, `prompts/induction_failure.txt`) so failure memories capture *what to avoid* and success memories capture *what to repeat*.

The induction LLM returns a JSON array; entries missing any of `title`/`description`/`content` are dropped silently.

### 3. Merge

For each candidate, `fleet.memory.merge.integrate_candidate(bank, candidate)` searches the existing bank for the 3 nearest neighbors (filtered by scope) and picks one of four actions based on cosine similarity to the closest neighbor:

| Similarity | Action | What happens |
|---|---|---|
| `> merge_thresholds[1]` (default `0.92`) | **merge / replace** | The induction LLM is asked to consolidate the two items, or to replace the old one outright. |
| `> merge_thresholds[0]` (default `0.75`) | **link** | Both items are kept; their `related_ids` are cross-linked. |
| otherwise | **insert** | The candidate becomes a new memory. |

If the merge LLM returns `keep_both`, the candidate falls through to the link branch.

You can tune the thresholds at construction:

```python
bank = ReasoningBank(merge_thresholds=(0.7, 0.9))
```

### Scheduler integration

When you call `graph.run(...)`, the scheduler walks every node, finds every unique `memory_bank` reachable through agent owners, and fires `bank.ingest_trajectory(...)` once per bank. By default this runs as a background `asyncio.Task` (`async_writeback=True`) so the user-visible run latency is unchanged. Banks without `judge_llm` and `induction_llm` configured are silently skipped — perfect for retrieval-only deployments.

---

## MaTTS (Memory-Aware Test-Time Scaling)

`matts_run(agent, task, bank=None, k=3, contrast_llm=None)` runs the same task `k` times in parallel through your compiled graph (or any runnable with an async `run(GraphState)`), then asks a **contrast LLM** to compare all `k` trajectories and distill higher-quality memories than any single rollout could produce.

```python
from fleet.memory import matts_run

states, distilled = await matts_run(graph, goal, bank=bank, k=3)
```

Contrast memories are stored with `source="matts_contrast"` and pass through the normal merge pipeline. The contrast LLM defaults to `bank.induction_llm`; override it via `contrast_llm=` if you want a different model for the contrast step (e.g., a stronger model for the harder synthesis task).

### When to use MaTTS

- **Bootstrapping a new scope.** Early runs are when memory quality matters most — a great memory you build today is reused on every subsequent run.
- **High-stakes or one-shot tasks.** Pay 3-5x the tokens to get a more reliable trajectory *and* better memories.
- **Sparse signal.** If trajectories vary widely (multiple plausible search strategies, etc.), the contrast prompt is much better at spotting the durable pattern than the per-trajectory inducer.

Cost scales linearly with `k`. For routine production use after a scope is well-populated, the standard (single-trajectory) writeback is usually enough.

---

## Memory API & UI

When `fleet ui` is running, the dashboard's **Memory** tab is wired to these routes (mounted under `/api`):

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/memory` | List memories. With `?q=...&k=20[&scope=...]`, runs semantic search. |
| `POST` | `/api/memory` | Manually add a memory (`{title, description, content, scope?}`). |
| `GET` | `/api/memory/{id}` | Fetch one. |
| `PUT` | `/api/memory/{id}` | Update title/description/content/confidence/scope. Re-embeds if any text field changed. |
| `DELETE` | `/api/memory/{id}` | Remove one. |
| `POST` | `/api/memory/export` | Returns `{items, count}` — full JSON dump, embeddings included. |
| `POST` | `/api/memory/import` | Accepts `{items: [...]}`, upserts on duplicate `id`. |

The bank used by the API is a module-level singleton (`fleet.server.memory_routes._BANK`) constructed lazily on the first request — same defaults as `ReasoningBank()`. To install a custom bank (e.g., a test bank with a stubbed embedder), call `fleet.server.memory_routes.set_bank(my_bank)`.

The wire format strips the `embedding` field from responses; export keeps it.

---

## Configuration reference

### `ReasoningBank(...)`

| Arg | Default | Notes |
|---|---|---|
| `store` | `SQLiteVecStore()` (or `InMemoryStore` if `sqlite-vec` is missing) | Anything implementing the `MemoryStore` protocol. |
| `embedder` | `MiniLMEmbedder()` | Must match the store's `dim`. |
| `scope` | `"global"` | Default scope for retrieval and ingestion. Per-call overrides allowed. |
| `judge_llm` | `None` | Required for `outcome=None` ingestion (auto-judge). |
| `induction_llm` | `None` | Required for ingestion; doubles as merge LLM and MaTTS contrast LLM. |
| `merge_thresholds` | `(0.75, 0.92)` | `(link_threshold, merge_threshold)`. |
| `async_writeback` | `True` | If True, scheduler writeback runs as a background task. Set False in tests for deterministic completion. |

### Environment variables

| Var | Default | Effect |
|---|---|---|
| `FLEET_MEMORY_DB` | `~/.fleet/reasoning_bank.db` | Path to the SQLite database. |

### `Agent(...)` memory args

| Arg | Default | Notes |
|---|---|---|
| `memory_bank` | `None` | Attach a `ReasoningBank` to enable retrieval + scheduler writeback. |
| `memory_k` | `5` | Top-k memories retrieved and prepended as a system message. |

---

## FAQ

**Does enabling memory change my graph?** No. The bank is opt-in per agent (via `memory_bank=`). Agents without a bank behave exactly as in v0.1.x. The scheduler hook is a no-op for banks that don't have an induction LLM set.

**What happens on a brand-new bank?** `retrieve()` returns `[]` and the agent runs with no prepended memory block. After the first run, the scheduler kicks off ingestion in the background.

**Can I run memory entirely offline?** Yes. `MiniLMEmbedder` runs locally. `SQLiteVecStore` is on-disk. The only network calls are the judge / induction / merge LLM completions — replace those with a local backend via polyrt (Ollama, MLX, etc.) and the whole loop is local.

**Will writeback slow my runs?** No. With `async_writeback=True` (default) the writeback is scheduled as a background `asyncio.Task` after the scheduler returns. In tests, set `async_writeback=False` and `await` the returned task explicitly.

**How big does the bank get?** Each `MemoryItem` is on the order of a few KB on disk (text fields + a 384-d float32 vector). For typical agent workloads the bank stays well under 100 MB even after thousands of trajectories.

**How do I share a bank across processes?** Point each process at the same `FLEET_MEMORY_DB`. SQLite serializes writes; concurrent retrievers are fine. For higher-concurrency setups, swap in a different `MemoryStore` (e.g., a Redis- or Postgres-backed one — implement the protocol).

**How do I wipe a bank?** Delete the SQLite file at `$FLEET_MEMORY_DB` (default `~/.fleet/reasoning_bank.db`). For per-scope wipes, use the API (`DELETE /api/memory/{id}` per item) or `await bank.delete(id)`.

**How do I export memories to another machine?** `await bank.export(path)` writes JSONL (one item per line, embeddings included). `await bank.import_(path)` reads them back, recomputing embeddings only for items missing one. The HTTP `/api/memory/export` and `/api/memory/import` routes do the same over the wire.

**Where's the paper?** Ouyang et al., *ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory*, ICLR 2026 — https://arxiv.org/abs/2509.25140
