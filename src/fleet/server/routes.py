from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fleet.core.graph import CompiledGraph, Graph
from fleet.core.scheduler import EventBus
from fleet.core.state import GraphState

router = APIRouter()

# ---------------------------------------------------------------------------
# In-process run store
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    run_id: str
    state: GraphState
    status: str  # running | paused | done | killed | error
    event_bus: EventBus
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

async def _execute_run(run_id: str, graph_module: str, backend: str) -> None:
    record = _RUNS[run_id]
    # Provider credentials are read from the process environment
    # (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY). Export them before
    # launching `fleet ui` — the UI does not accept keys.
    try:
        raw = load_graph(graph_module)
        cg: CompiledGraph = raw.compile(backend=backend) if isinstance(raw, Graph) else raw
        await _drive_cg(record, run_id, cg)
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


async def _execute_run_from_spec(run_id: str, cg: CompiledGraph) -> None:
    """Run a pre-compiled CompiledGraph (built from a spec) end-to-end."""
    record = _RUNS[run_id]
    try:
        await _drive_cg(record, run_id, cg)
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


async def _drive_cg(record: RunRecord, run_id: str, cg: CompiledGraph) -> None:
    cg._event_bus = record.event_bus
    state = await cg.run(record.state)
    record.state = state
    record.status = "done"
    await record.event_bus.emit({"type": "run_finished", "run_id": run_id, "status": "done"})


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    graph_module: str
    goal: str
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
    )
    _RUNS[run_id] = record
    record.task = asyncio.create_task(_execute_run(run_id, body.graph_module, body.backend))
    return {"run_id": run_id}


# ---------------------------------------------------------------------------
# Builder endpoints — run / export a spec authored in the visual builder
# ---------------------------------------------------------------------------

class FromSpecRequest(BaseModel):
    spec: dict[str, Any]
    goal: str
    backend: str = "sqlite"


class ExportPythonRequest(BaseModel):
    spec: dict[str, Any]


@router.post("/runs/from-spec", status_code=201)
async def create_run_from_spec(body: FromSpecRequest) -> dict[str, str]:
    """Validate and run a GraphSpec authored in the visual builder.

    The spec is untrusted input — it is validated through Pydantic and
    constructed using the same loader the CLI uses. No arbitrary Python is
    exec'd; conditions are looked up by name in the registry; tool names
    must already be registered.
    """
    from fleet.graphspec.loader import load_graph_spec

    try:
        cg = load_graph_spec(body.spec, backend=body.backend)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid spec: {exc}") from exc

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    event_bus = EventBus()
    state = GraphState(
        goal=body.goal,
        metadata={
            "run_id": run_id,
            "graph_source": "builder",
            "graph_name": str(body.spec.get("name", "")),
        },
    )
    record = RunRecord(
        run_id=run_id,
        state=state,
        status="running",
        event_bus=event_bus,
    )
    _RUNS[run_id] = record
    record.task = asyncio.create_task(_execute_run_from_spec(run_id, cg))
    return {"run_id": run_id}


@router.post("/export-python")
async def export_python(body: ExportPythonRequest) -> dict[str, str]:
    """Render a GraphSpec as a runnable Python source file."""
    from fleet.graphspec import GraphSpec, spec_to_python

    try:
        spec = GraphSpec.model_validate(body.spec)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid spec: {exc}") from exc
    return {"source": spec_to_python(spec)}


@router.get("/conditions")
async def list_conditions() -> dict[str, Any]:
    """Return the names of edge conditions the loader will accept.

    Parametric conditions are listed with a trailing ``:<key>`` placeholder
    so the UI can render them as an input.
    """
    from fleet.graphspec.conditions import _CONDITION_REGISTRY, _PARAMETRIC_REGISTRY

    plain = sorted(_CONDITION_REGISTRY.keys())
    parametric = [f"{p}:<key>" for p in sorted(_PARAMETRIC_REGISTRY.keys())]
    return {"conditions": plain, "parametric": parametric}


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    """Return the names of registered fleet tools so the UI can multi-select."""
    from fleet.tools.base import _TOOL_REGISTRY

    return {"tools": sorted(_TOOL_REGISTRY.keys())}


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


@router.get("/providers")
async def list_providers() -> dict[str, Any]:
    """Return the canonical set of providers the loader supports, plus per-
    provider hints the UI uses to render the configuration panel."""
    from fleet.graphspec.spec import SUPPORTED_PROVIDERS
    from fleet.providers.client import _DEFAULT_BASE_URLS

    providers = sorted(SUPPORTED_PROVIDERS)
    details = [
        {
            "name": name,
            "requires_base_url": name == "custom",
            "default_base_url": _DEFAULT_BASE_URLS.get(name),
        }
        for name in providers
    ]
    return {"providers": providers, "details": details}
