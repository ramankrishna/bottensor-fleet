# bottensor-fleet v0.2.0 — Gap Test Report

**Branch:** v0.2-reasoning-bank
**Wheel:** bottensor_fleet-0.2.0-py3-none-any.whl (180KB)
**Date:** 2026-05-27
**Tester:** Dispatch (automated) + Ram (manual UI)

## Phase 0 — Environment Setup
- **Status:** PASS
- Installed wheel in isolated venv at /tmp/fleet-gap-test/.venv
- Set ANTHROPIC_API_KEY

## Phase 1 — Fleet Replay E2E (4 subtests)
All PASS:
1. `fleet replay` basic run — completed successfully
2. `fleet replay --provider anthropic` — works with new alias
3. `fleet replay` output format — JSON output valid
4. Replay result consistency — deterministic across runs

## Phase 2 — Backward Compatibility (4 subtests)
All PASS:
1. `fleet run` basic execution — works
2. CLI `--version` flag — prints 0.2.0
3. Import `fleet.memory` — module loads correctly
4. Import `fleet.memory.stores.InMemoryStore` — exports work (was broken, fixed pre-gap-test)

## Phase 3 — Memory Tab UI (9 subtests, manual click-through by Ram)
All PASS (verified by user):
1. Memory tab loads without errors
2. 3 seeded items display correctly
3. Search/filter works
4. Item detail view opens
5. Edit functionality works
6. Delete functionality works
7. Export works
8. Import works
9. UI responsive/no console errors

**Note:** Initial black screen caused by infinite re-render bug in ProviderPicker.tsx (not MemoryTab.tsx). The `/api/providers` response was `{providers: [...]}` but code expected a flat array. Fixed by normalizing: `setProviders(Array.isArray(data) ? data : data.providers ?? [])`. Fix applied to working tree, not yet committed.

## Known Issues (flagged, not blocking)
1. `learning_research_team.py` example — broken with claude-sonnet-4-6 (assistant message prefill not supported)
2. `fleet.__version__` — module-level attribute missing (CLI --version works fine)
3. `solo_agent.py` — uses "claude" provider name instead of "anthropic"
4. `fleet run` — doesn't print final output to stdout
5. ProviderPicker.tsx fix — in working tree, needs commit on v0.2-reasoning-bank branch

## Verdict
**HOLD** — User explicitly requested no publish. All tests pass. Package is release-ready pending user go-ahead.
