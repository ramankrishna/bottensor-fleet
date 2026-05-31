# Changelog

## 0.3.0 (2026-05-31)

### Added
- **JSON graph format (`GraphSpec`)** — a declarative, first-class graph format that the rest of fleet treats on par with `.py` graphs. The shapes are Pydantic models in `fleet.graphspec`: `GraphSpec`, `NodeSpec`, `AgentSpec`, `EdgeSpec`, `Position`. Strict validation: unknown providers, unknown tool names, dangling edges, missing entry/exit, duplicate node ids, and unknown condition names all raise clear `ValidationError`s.
- **`load_graph_spec(spec | dict | json_str | path)`** — turns a JSON spec into a runnable `CompiledGraph` by reusing the existing FleetLLM / Agent / Graph runtime.
- **`spec_to_python(spec)`** — round-trips a spec back to a hand-readable Python file that mirrors the bundled-examples style. Generated source passes `ruff check`. A regression suite (`tests/test_graphspec_roundtrip.py`) locks in that the loader path and the export-then-run path produce structurally + behaviourally equivalent graphs.
- **Condition registry** — `fleet.graphspec.register_condition(name, fn)` and `register_parametric_condition(prefix, factory)`. JSON edges can only reference registered names; no `eval`, no arbitrary code. Built-ins: `always`, `max_steps_not_hit`, and the parametric `scratchpad_true:<key>`.
- **CLI:** `fleet run graph.json --goal "…"` and `fleet export graph.json -o graph.py`.
- **Visual graph builder** — new **Builder** tab in `fleet ui`. Drag Agent nodes from a palette onto a ReactFlow canvas; click to configure provider / model / system / tools / memory / `base_url` (shown only when the provider needs it) / entry+exit pills; click edges to set a registry condition. Live status badges overlay the same canvas via the existing WebSocket stream. Toolbar: Run, Export .py, Save .json, Open .json.
- **Server endpoints (untrusted spec path):**
  - `POST /api/runs/from-spec` — validates a browser-authored spec, compiles, runs in the background; 422 on any validation error.
  - `POST /api/export-python` — server-side rendering of a spec to Python source.
  - `GET /api/conditions` — registry contents (plain + parametric).
  - `GET /api/tools` — registered `@tool` names for the agent-tools multiselect.
- **New providers** — `deepseek` (OpenAI-compatible, default base URL `https://api.deepseek.com`, `DEEPSEEK_API_KEY` → `OPENAI_API_KEY` fallback) and `custom` (OpenAI-compatible, required `base_url`, `CUSTOM_API_KEY` → `OPENAI_API_KEY` fallback). Covers vLLM, Together, Ollama-as-OpenAI, LM Studio. Implemented fleet-side via the OpenAI SDK + `base_url` override; no polyrt release required.
- **SSRF guard** — `fleet.graphspec.validate_base_url()` blocks non-http(s) schemes (`file://`, `gopher://`, …) and known cloud-metadata endpoints (AWS `169.254.169.254`, AWS ECS `169.254.170.2`, GCP `metadata.google.internal` / `metadata.goog`, Alibaba `100.100.100.200`, Azure IMDS `169.254.169.253`). Localhost / RFC1918 are deliberately allowed for local vLLM / Ollama. Wired into `AgentSpec` so every code path (CLI, loader, `POST /api/runs/from-spec`) gets the same protection.
- **`fleet examples`** lists and extracts JSON specs as well as `.py` files. `examples/research_team.json` (planner → researcher → writer with `web_search` / `web_fetch`) ships in the wheel as a reference.
- **Documentation:** [`docs/visual-builder.md`](docs/visual-builder.md) — using the builder, GraphSpec field reference, trust model.

### Changed
- `/api/providers` returns the canonical list of supported providers plus per-provider UI hints (`requires_base_url`, `default_base_url`), instead of polling polyrt's entry-point registry.
- `FleetLLM` routes `openai` / `deepseek` / `custom` through the OpenAI SDK directly (with `base_url` override) for both tool and no-tool calls; Anthropic / Claude unchanged.

### Security
- Conditions in JSON specs are whitelist-only — there is no `eval` path; an injected name like `__import__('os').system(...)` is rejected at validation with an "unknown condition" error.
- Tool names in JSON specs are whitelist-only via the `@tool` registry.
- `base_url` SSRF guard (described above).
- The Builder UI shows a ⚠ warning under the tools picker when `python_exec` is selected, noting that it runs unsandboxed.

### Compatibility
- No breaking changes to v0.2 APIs. Existing `.py` graphs, the `fleet run` CLI for them, the Memory tab, and ReasoningBank / MaTTS all behave identically. The JSON format and the Builder tab are purely additive.

## 0.2.0 (2026-05-27)

### Added
- **ReasoningBank** — experience-augmented memory for agents ([Ouyang et al., ICLR 2026](https://arxiv.org/abs/2509.25140)). Public API: `fleet.ReasoningBank`, `fleet.MemoryItem`, `fleet.memory.matts_run`, plus `fleet.memory.{stores, embedders, get_embedder, register_embedder}`.
  - Retrieval: top-k cosine search over MiniLM embeddings, scope-filtered, prepended to the agent's context as a system message.
  - Ingestion pipeline: LLM judge → success/failure induction → merge (replace / merge / link / insert) based on cosine similarity thresholds.
  - Scheduler integration: every run with a memory-aware agent triggers an async writeback once per unique bank reachable from the graph. Banks without `induction_llm` + `judge_llm` are silently skipped (retrieval-only mode).
  - Export / import as JSONL (`bank.export(path)` / `bank.import_(path)`), with embeddings preserved.
- **MaTTS (Memory-Aware Test-Time Scaling)** — `matts_run(agent, task, bank, k=3)` runs k parallel rollouts and contrast-distills higher-quality memories than any single rollout could produce. Stored with `source="matts_contrast"`.
- **Storage backends:**
  - `SQLiteVecStore` (default) — persistent, uses the `sqlite-vec` extension. DB path configurable via `$FLEET_MEMORY_DB` (default `~/.fleet/reasoning_bank.db`).
  - `InMemoryStore` — pure-Python, no dependencies, used in tests and as a fallback when `sqlite-vec` isn't installed.
- **Embedders:** `minilm` (default, 384-d via `sentence-transformers`), `mlx` (Apple Silicon), `polyrt` (any polyrt backend that exposes embeddings). Custom embedders registerable via `fleet.memory.register_embedder(name, factory)`.
- **Memory UI:** new **Memory** tab in `fleet ui` — browse, search, edit, delete, manually add, and export / import memories. Backed by `/api/memory*` routes.
- **Examples:** `fleet examples learning_research_team` (4-agent team sharing a bank) and `fleet examples matts_solo` (single agent with MaTTS k=3).
- **Agent memory args:** `Agent(memory_bank=..., memory_k=5)` enables retrieval + writeback on a per-agent basis.
- New documentation: [`docs/memory.md`](docs/memory.md) — comprehensive guide to ReasoningBank, MaTTS, storage, embedders, the induction pipeline, API routes, configuration reference, and FAQ.

### Changed
- `polyrt>=0.1.1` is now the minimum (was `>=0.1.0`) — required for the alias and embedding hooks used by the `polyrt` embedder.

### Compatibility
- No breaking changes. The bank is fully opt-in: existing graphs and agents run unchanged. The scheduler's memory writeback is a no-op for agents without `memory_bank` set, and for banks without `induction_llm` / `judge_llm` configured.
- `[memory]` is a new optional extra (`sqlite-vec`, `sentence-transformers`). Without it, importing `fleet.memory` still works; the bank falls back to `InMemoryStore` and raises on instantiation if no embedder is available.

## 0.1.2 (2026-05-27)

### Fixed
- `fleet replay <run_id>` now re-executes the original graph using its absolute source path persisted at run time
- Scheduler now emits a `logger.warning` when a run terminates by `max_steps`, making unbounded cycles easier to spot

### Added
- Examples ship inside the wheel (`src/fleet/examples/`) and can be listed or extracted via `fleet examples [name]`
- `state.metadata["fleet_version"]` recorded on every checkpoint for debugging old runs after upgrades
- `state.metadata["replayed_from"]` stamped on replayed runs so lineage is recoverable

### Upstream
- Filed [polyrt#1](https://github.com/ramankrishna/polyRT/issues/1) requesting `anthropic` as an alias for the `claude` backend

## 0.1.1 (2026-05-05)

### Fixed
- LLM calls now work out of the box: `polyrt[anthropic,openai]` is a runtime dep
- `fleet add-agent` generates correct backend names (`claude` not `anthropic`)
- `fleet ui` actually serves the bundled React UI
- `fleet replay` persists and reads graph source from checkpoint metadata
- `Agent.step` auto-seeds `state.goal` as first user message when none exist
- Provider errors now surface as one-line `ProviderError`, not 30-line tracebacks
- `/api/providers` returns only actually-registered backends
- Scheduler sets `state.metadata['terminated_by']` so silent max_steps termination is detectable

### Added
- `fleet --version`
- `fleet ls` now shows `saved_at` and `goal`, not just run_id
- `bottensor-fleet[search]` and `bottensor-fleet[redis]` extras
- Improved `fleet new` scaffold demonstrates real Agent usage

### Changed
- `duckduckgo-search` and `redis` moved to optional extras (smaller default install)

## 0.1.0 (2026-05-05)

Initial release. Internal dry run; not announced. Use 0.1.1 or later.
