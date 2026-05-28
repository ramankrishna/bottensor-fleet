"""Memory API routes — CRUD, search, export, import for ReasoningBank."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fleet.memory import ReasoningBank
from fleet.memory.item import MemoryItem

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared bank — lazily constructed on first request
# ---------------------------------------------------------------------------

_BANK: ReasoningBank | None = None


def get_bank() -> ReasoningBank:
    global _BANK
    if _BANK is None:
        _BANK = ReasoningBank()
    return _BANK


def set_bank(bank: ReasoningBank | None) -> None:
    """Test hook: install a specific bank (or None to reset)."""
    global _BANK
    _BANK = bank


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class MemoryUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    content: str | None = None
    confidence: float | None = None
    scope: str | None = None


class MemoryCreate(BaseModel):
    title: str
    description: str
    content: str
    scope: str | None = None


class ImportPayload(BaseModel):
    items: list[dict[str, Any]]


def _serialize(item: MemoryItem) -> dict[str, Any]:
    """Strip the embedding from the wire format — clients don't need it."""
    data = item.model_dump(mode="json")
    data.pop("embedding", None)
    return data


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

@router.get("/memory")
async def list_memory(q: str | None = None, k: int = 20, scope: str | None = None) -> dict[str, Any]:
    """List memories. If ``q`` is provided, return semantic search results."""
    bank = get_bank()
    if q:
        results = await bank.retrieve(q, k=k, scope=scope, min_confidence=0.0)
        return {"items": [_serialize(it) for it in results], "query": q}
    items = bank.store.list_all(scope=scope)
    return {"items": [_serialize(it) for it in items]}


@router.post("/memory", status_code=201)
async def create_memory(body: MemoryCreate) -> dict[str, Any]:
    bank = get_bank()
    item = await bank.add_manual(
        title=body.title,
        description=body.description,
        content=body.content,
        scope=body.scope,
    )
    return _serialize(item)


@router.get("/memory/{item_id}")
async def get_memory(item_id: str) -> dict[str, Any]:
    bank = get_bank()
    item = bank.store.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Memory '{item_id}' not found")
    return _serialize(item)


@router.put("/memory/{item_id}")
async def update_memory(item_id: str, body: MemoryUpdate) -> dict[str, Any]:
    bank = get_bank()
    item = bank.store.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Memory '{item_id}' not found")

    changed = False
    if body.title is not None:
        item.title = body.title
        changed = True
    if body.description is not None:
        item.description = body.description
        changed = True
    if body.content is not None:
        item.content = body.content
        changed = True
    if body.confidence is not None:
        item.confidence = body.confidence
    if body.scope is not None:
        item.scope = body.scope

    if changed:
        item.embedding = bank.embedder.embed([item.signature_text()])[0]
    bank.store.update(item)
    return _serialize(item)


@router.delete("/memory/{item_id}")
async def delete_memory(item_id: str) -> dict[str, str]:
    bank = get_bank()
    if bank.store.get(item_id) is None:
        raise HTTPException(status_code=404, detail=f"Memory '{item_id}' not found")
    bank.store.delete(item_id)
    return {"status": "deleted", "id": item_id}


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------

@router.post("/memory/export")
async def export_memory() -> dict[str, Any]:
    """Return all memories as a JSON array (in-line, not a download)."""
    bank = get_bank()
    items = bank.store.list_all()
    return {"items": [it.model_dump(mode="json") for it in items], "count": len(items)}


@router.post("/memory/import")
async def import_memory(body: ImportPayload) -> dict[str, Any]:
    """Import items from a JSON payload. Upserts on duplicate id."""
    bank = get_bank()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for raw in body.items:
            f.write(json.dumps(raw) + "\n")
        tmp_path = Path(f.name)
    try:
        n = await bank.import_(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"imported": n}
