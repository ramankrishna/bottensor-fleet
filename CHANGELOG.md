# Changelog

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
