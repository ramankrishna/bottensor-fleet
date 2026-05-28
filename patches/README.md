# Fleet v0.2.0 Patches

These files are patched versions of the `bottensor-fleet` v0.2.0 package.
Copy them into your `.venv/lib/python3.11/site-packages/fleet/` to apply fixes.

## Fixes Applied

### 1. agent.py → `fleet/agents/agent.py`
- **Agent handoff bug**: When a prior node leaves an assistant message at the tail, the agent now injects a user message with its goal before calling the LLM. Fixes `invalid_request_error: conversation must end with a user message`.
- **Scratchpad resolution**: References like `scratchpad['q1']` in agent goals are resolved to actual values, so fan-out agents (researcher_a, researcher_b) get distinct prompts.
- **Event emission**: Agents now emit `tool_call`, `tool_result`, and `state_update` events via the scheduler's EventBus, populating the Activity Log in the UI.

### 2. routes.py → `fleet/server/routes.py`
- **API key passthrough**: `_execute_run` now injects `provider_keys` from the request body into environment variables (`ANTHROPIC_API_KEY`, etc.) before running the graph, and restores them after.

### 3. scheduler.py → `fleet/core/scheduler.py`
- **node_start events**: Emits `node_start` before each node runs (UI shows nodes as "running").
- **Event bus registry**: Module-level `_EVENT_BUS_REGISTRY` keyed by run_id so agents can look up the bus without passing non-serializable objects through GraphState.
- **Memory writeback fix**: `_run_memory_writebacks` now scans function globals/closures for Agent objects with `memory_bank`, not just bound methods. Fixes writeback for graphs using plain wrapper functions.

### 4. Frontend JS (index-Cw810M4y.js)
- **API response parsing**: Fixed provider and graphs fetch to unwrap `{providers: [...]}` / `{graphs: [...]}` responses instead of storing the whole object (caused React Error #185 infinite re-render / black screen).
- **provider_keys in POST**: Start button now sends `provider_keys` in the request body.
- **state_updated handler**: Activity log now renders LLM text responses.
