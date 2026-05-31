from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .memory_routes import router as memory_router
from .routes import router
from .ws import ws_router


def _check_sqlite_vec() -> None:
    """Emit a visible warning when sqlite-vec is missing — memory will be in-process only."""
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        print(
            "⚠ sqlite_vec not installed — memory will use InMemoryStore (not persistent). "
            "Install with: pip install sqlite-vec",
            flush=True,
        )


def create_app() -> FastAPI:
    _check_sqlite_vec()
    app = FastAPI(title="bottensor-fleet", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.include_router(router, prefix="/api")
    app.include_router(memory_router, prefix="/api")
    app.include_router(ws_router)
    static_dir = _find_static_dir()
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    return app


def _find_static_dir() -> Path | None:
    """Locate the built UI assets.

    Wheel installs ship them at ``src/fleet/static`` (see pyproject force-include).
    For editable / source checkouts, fall back to ``ui/dist`` at the repo root.
    """
    packaged = Path(__file__).parent.parent / "static"
    if packaged.exists():
        return packaged
    repo_dist = Path(__file__).resolve().parents[3] / "ui" / "dist"
    if repo_dist.exists():
        return repo_dist
    return None
