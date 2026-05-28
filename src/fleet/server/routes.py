from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fleet.core.graph import CompiledGraph, Graph
from fleet.core.scheduler import EventBus
from fleet.core.state import GraphState

router = APIRouter()

# ---------------------------------------------------------------------------
# In-process run store (provider_keys wiped on completion)
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str
    state: GraphState
    status: str  # running | paused | done | killed | error
    event_bus: EventBus
    provider_keys: dict[str, str] = field(default_factory=dict)
    task: asyncio.Task | None = None
    error: str | None = None


_RUNS: dict[str, RunRecord] = {}


def _get_run(run_id: str) -> RunRecord | None:
    return _RUNS.get(run_id)


# ---------------------------------------------------------------------------
# Graph loader (file path or dotted module name)
# ---------------------------------------------------------------------------

def load_graph(graph_module: str) -> Graph | CompiledGraph:
    if graph_module.endswith(".py") or "/" in graph_module or os.sep in graph_module:
        path = os.path.abspath(graph_module)
        spec = importlib.util.spec_from_file_location("_fleet_dyn", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load: {graph_module!r}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        mod = importlib.import_module(graph_module)

    if hasattr(mod, "create_graph"):
        return mod.create_graph()  # type: ignore[no-any-return]
    if hasattr(mod, "graph"):
        return mod.graph  # type: ignore[no-any-return]
    raise ValueError(
        f"Module {graph_module!r} must export 'graph: Graph' or 'create_graph() -> Graph'"
    )


# ---------------------------------------------------------------------------
# Background run execution
# ---------------------------------------------------------------------------

_KEY_ENV_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mlx": "MLX_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}


async def _execute_run(run_id: str, graph_module: str, backend: str) -> None:
    record = _RUNS[run_id]
    # Inject provider keys into env so polyrt backends can find them
    _saved_env: dict[str, str | None] = {}
    for provider, key in record.provider_keys.items():
        if key:
            env_var = _KEY_ENV_MAP.get(provider.lower(), f"{provider.upper()}_API_KEY")
            _saved_env[env_var] = os.environ.get(env_var)
            os.environ[env_var] = key
    try:
        raw = load_graph(graph_module)
        cg: CompiledGraph = raw.compile(backend=backend) if isinstance(raw, Graph) else raw
        # Wire the run's EventBus so WebSocket subscribers receive events.
        cg._event_bus = record.event_bus

        state = await cg.run(record.state)
        record.state = state
        record.status = "done"
        await record.event_bus.emit({"type": "run_finished", "run_id": run_id, "status": "done"})
    except asyncio.CancelledError:
        record.status = "killed"
        await record.event_bus.emit({"type": "run_finished", "run_id": run_id, "status": "killed"})
        raise
    except Exception as exc:
        record.status = "error"
        record.error = str(exc)
        await record.event_bus.emit(
            {"type": "run_finished", "run_id": run_id, "status": "error", "error": str(exc)}
        )
    finally:
        # Restore original env vars
        for env_var, old_val in _saved_env.items():
            if old_val is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = old_val
        record.provider_keys.clear()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    graph_module: str
    goal: str
    provider_keys: dict[str, str] | None = None
    backend: str = "sqlite"


# ---------------------------------------------------------------------------
# Run lifecycle endpoints
# ---------------------------------------------------------------------------

@router.post("/runs", status_code=201)
async def create_run(body: RunRequest) -> dict[str, str]:
    """Start a new graph run and return its run_id."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    event_bus = EventBus()
    state = GraphState(
        goal=body.goal,
        metadata={"run_id": run_id, "graph_module": body.graph_module},
    )
    record = RunRecord(
        run_id=run_id,
        state=state,
        status="running",
        event_bus=event_bus,
        provider_keys=dict(body.provider_keys or {}),
    )
    _RUNS[run_id] = record
    record.task = asyncio.create_task(_execute_run(run_id, body.graph_module, body.backend))
    return {"run_id": run_id}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Return the current state of a run."""
    record = _RUNS.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {
        "run_id": record.run_id,
        "status": record.status,
        "goal": record.state.goal,
        "message_count": len(record.state.messages),
        "scratchpad_keys": list(record.state.scratchpad.keys()),
        "error": record.error,
    }


@router.post("/runs/{run_id}/pause")
async def pause_run(run_id: str) -> dict[str, str]:
    """Cancel the run task; the last checkpoint is preserved for resume."""
    record = _RUNS.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if record.task and not record.task.done():
        record.task.cancel()
        try:
            await record.task
        except (asyncio.CancelledError, Exception):
            pass
    record.status = "paused"
    return {"status": record.status}


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: str) -> dict[str, str]:
    """Re-start a paused run from its last checkpoint."""
    record = _RUNS.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if record.status != "paused":
        raise HTTPException(status_code=409, detail=f"Run is '{record.status}', not paused")
    graph_module = record.state.metadata.get("graph_module", "")
    record.status = "running"
    record.task = asyncio.create_task(_execute_run(run_id, graph_module, "sqlite"))
    return {"status": record.status}


@router.post("/runs/{run_id}/kill")
async def kill_run(run_id: str) -> dict[str, str]:
    """Hard-cancel a running task."""
    record = _RUNS.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if record.task and not record.task.done():
        record.task.cancel()
        try:
            await record.task
        except (asyncio.CancelledError, Exception):
            pass
    record.status = "killed"
    return {"status": record.status}


# ---------------------------------------------------------------------------
# Discovery endpoints
# ---------------------------------------------------------------------------

@router.get("/graphs")
async def list_graphs() -> dict[str, Any]:
    """List .py graph files in ./graphs/ as paths relative to CWD."""
    graphs_dir = "graphs"
    if not os.path.isdir(graphs_dir):
        return {"graphs": [], "directory": graphs_dir}
    files = sorted(
        os.path.join(graphs_dir, f)
        for f in os.listdir(graphs_dir)
        if f.endswith(".py") and not f.startswith("_")
    )
    return {"graphs": files, "directory": graphs_dir}


# TODO: replace with dynamic polyrt registry once entry-points are populated
_KNOWN_BACKENDS = ["anthropic", "openai", "mlx", "ollama"]


@router.get("/providers")
async def list_providers() -> dict[str, Any]:
    """Return actually-registered polyrt backend names."""
    try:
        from polyrt import registry as _reg
        _reg._load_entry_points()
        registered = list(_reg._FACTORIES.keys())
    except Exception:
        registered = []
    backends = registered if registered else _KNOWN_BACKENDS
    return {"providers": sorted(set(backends))}
