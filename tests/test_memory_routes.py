"""Tests for the /api/memory FastAPI routes."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fleet.memory import ReasoningBank
from fleet.memory.stores.inmemory import InMemoryStore
from fleet.server.app import create_app
from fleet.server.memory_routes import set_bank


class _StaticEmbedder:
    DIM = 4

    @property
    def dim(self) -> int:
        return self.DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            for i, ch in enumerate(text.lower()):
                vec[(ord(ch) + i) % self.DIM] += 1.0
            n = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / n for v in vec])
        return out


@pytest.fixture
def client():
    bank = ReasoningBank(store=InMemoryStore(), embedder=_StaticEmbedder())
    set_bank(bank)
    app = create_app()
    with TestClient(app) as c:
        yield c
    set_bank(None)


def test_list_memory_empty(client):
    r = client.get("/api/memory")
    assert r.status_code == 200
    assert r.json() == {"items": []}


def test_create_then_list_memory(client):
    r = client.post(
        "/api/memory",
        json={"title": "Tip A", "description": "desc", "content": "body"},
    )
    assert r.status_code == 201
    created = r.json()
    assert created["title"] == "Tip A"
    assert created["id"].startswith("mem_")
    assert "embedding" not in created  # embedding stripped from wire

    r2 = client.get("/api/memory")
    assert r2.status_code == 200
    items = r2.json()["items"]
    assert len(items) == 1
    assert items[0]["title"] == "Tip A"


def test_get_memory_404(client):
    r = client.get("/api/memory/mem_nope")
    assert r.status_code == 404


def test_get_memory(client):
    r = client.post(
        "/api/memory", json={"title": "T", "description": "d", "content": "c"}
    )
    item_id = r.json()["id"]
    r2 = client.get(f"/api/memory/{item_id}")
    assert r2.status_code == 200
    assert r2.json()["title"] == "T"


def test_update_memory(client):
    r = client.post(
        "/api/memory", json={"title": "old", "description": "d", "content": "c"}
    )
    item_id = r.json()["id"]

    r2 = client.put(
        f"/api/memory/{item_id}",
        json={"title": "new", "confidence": 0.5},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["title"] == "new"
    assert body["confidence"] == 0.5

    # confirm persistence
    r3 = client.get(f"/api/memory/{item_id}")
    assert r3.json()["title"] == "new"


def test_update_memory_404(client):
    r = client.put("/api/memory/mem_nope", json={"title": "x"})
    assert r.status_code == 404


def test_delete_memory(client):
    r = client.post(
        "/api/memory", json={"title": "x", "description": "d", "content": "c"}
    )
    item_id = r.json()["id"]
    r2 = client.delete(f"/api/memory/{item_id}")
    assert r2.status_code == 200
    assert r2.json()["status"] == "deleted"

    r3 = client.get(f"/api/memory/{item_id}")
    assert r3.status_code == 404


def test_delete_memory_404(client):
    r = client.delete("/api/memory/mem_nope")
    assert r.status_code == 404


def test_search_memory(client):
    client.post(
        "/api/memory",
        json={"title": "Python tips", "description": "py", "content": "use venv"},
    )
    client.post(
        "/api/memory",
        json={"title": "Rust gotchas", "description": "rs", "content": "ownership"},
    )
    r = client.get("/api/memory", params={"q": "python venv"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "python venv"
    assert len(body["items"]) >= 1


def test_export_then_import_round_trip(client):
    client.post(
        "/api/memory", json={"title": "A", "description": "d", "content": "c"}
    )
    client.post(
        "/api/memory", json={"title": "B", "description": "d", "content": "c"}
    )

    r = client.post("/api/memory/export")
    assert r.status_code == 200
    exported = r.json()
    assert exported["count"] == 2
    items = exported["items"]

    # Reset to a fresh bank, then import
    bank = ReasoningBank(store=InMemoryStore(), embedder=_StaticEmbedder())
    set_bank(bank)

    r2 = client.post("/api/memory/import", json={"items": items})
    assert r2.status_code == 200
    assert r2.json()["imported"] == 2

    r3 = client.get("/api/memory")
    titles = sorted(i["title"] for i in r3.json()["items"])
    assert titles == ["A", "B"]
