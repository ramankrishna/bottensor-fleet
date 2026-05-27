from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .memory_routes import router as memory_router
from .routes import router
from .ws import ws_router


def create_app() -> FastAPI:
    app = FastAPI(title="bottensor-fleet", version="0.1.0")
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
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    return app
