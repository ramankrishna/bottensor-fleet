# Changelog

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
