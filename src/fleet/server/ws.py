from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()

# Event types forwarded to connected clients:
#   node_started, node_finished, tool_called, tool_returned,
#   state_updated, run_finished, heartbeat, error


@ws_router.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    """Stream scheduler events for *run_id* as JSON frames over WebSocket."""
    # Lazy import to avoid import-time circularity between ws ↔ routes.
    from fleet.server.routes import _RUNS

    await websocket.accept()

    record = _RUNS.get(run_id)
    if record is None:
        await websocket.send_json({"type": "error", "message": f"Run '{run_id}' not found"})
        await websocket.close(code=1008)
        return

    # If the run already finished, send a single terminal frame and close.
    if record.status in ("done", "killed", "error"):
        await websocket.send_json(
            {
                "type": "run_finished",
                "run_id": run_id,
                "status": record.status,
                "error": record.error,
            }
        )
        return

    # Live run: subscribe a queue-backed callback to the EventBus.
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def _enqueue(event: dict[str, Any]) -> None:
        await queue.put(event)

    record.event_bus.subscribe(_enqueue)

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "run_id": run_id})
                continue

            # Re-map scheduler event types to the public event vocabulary.
            event = _normalize_event(event, run_id)
            await websocket.send_json(event)

            if event.get("type") == "run_finished":
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normalize_event(event: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Map internal scheduler events onto the public event schema."""
    evt_type = event.get("type", "")
    if evt_type == "node_complete":
        return {**event, "type": "node_finished", "run_id": run_id}
    if evt_type == "node_start":
        return {**event, "type": "node_started", "run_id": run_id}
    if evt_type == "tool_call":
        return {**event, "type": "tool_called", "run_id": run_id}
    if evt_type == "tool_result":
        return {**event, "type": "tool_returned", "run_id": run_id}
    if evt_type == "state_update":
        return {**event, "type": "state_updated", "run_id": run_id}
    # run_finished and heartbeat pass through unchanged.
    return {**event, "run_id": run_id}
