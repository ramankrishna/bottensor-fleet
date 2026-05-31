"""Tests for the builder-facing server endpoints.

- POST /api/runs/from-spec validates the spec and starts a run.
- POST /api/export-python renders a spec to Python source.
- GET  /api/conditions and /api/tools expose registry contents.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from fleet.core.messages import AgentMessage
from fleet.server import routes as _routes
from fleet.server.app import create_app


@pytest.fixture(autouse=True)
def _stub_llms(monkeypatch):
    """Stub FleetLLM.complete so no live providers are required."""
    from fleet.providers import client as _client

    async def _fake_complete(self, messages, tools=None):  # noqa: ARG001
        return AgentMessage(role="assistant", content="builder ok")

    monkeypatch.setattr(_client.FleetLLM, "complete", _fake_complete)
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-for-build")
    yield
    # Clear runs between tests so global state doesn't leak.
    _routes._RUNS.clear()


def _minimal_spec() -> dict:
    return {
        "version": "0.3",
        "name": "builder_team",
        "nodes": [
            {
                "id": "a",
                "type": "agent",
                "agent": {
                    "name": "a",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "system": "do thing",
                    "tools": [],
                    "memory_bank": False,
                },
            },
            {
                "id": "b",
                "type": "agent",
                "agent": {
                    "name": "b",
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "system": "do other thing",
                    "tools": [],
                    "memory_bank": False,
                },
            },
        ],
        "edges": [{"src": "a", "dst": "b"}],
        "entry": "a",
        "exit": "b",
    }


# ---------------------------------------------------------------------------
# /api/runs/from-spec
# ---------------------------------------------------------------------------

def test_from_spec_starts_run_and_returns_run_id():
    client = TestClient(create_app())
    r = client.post(
        "/api/runs/from-spec",
        json={"spec": _minimal_spec(), "goal": "hi"},
    )
    assert r.status_code == 201, r.text
    run_id = r.json()["run_id"]
    assert run_id.startswith("run_")


def test_from_spec_rejects_invalid_spec():
    """An untrusted spec with a bad provider must be rejected, not exec'd."""
    bad = _minimal_spec()
    bad["nodes"][0]["agent"]["provider"] = "not-a-provider"

    client = TestClient(create_app())
    r = client.post("/api/runs/from-spec", json={"spec": bad, "goal": "hi"})
    assert r.status_code == 422
    assert "Invalid spec" in r.text


def test_from_spec_rejects_dangling_edge():
    bad = _minimal_spec()
    bad["edges"].append({"src": "a", "dst": "ghost"})

    client = TestClient(create_app())
    r = client.post("/api/runs/from-spec", json={"spec": bad, "goal": "hi"})
    assert r.status_code == 422


def test_from_spec_rejects_unknown_condition():
    """Unknown conditions must be rejected — JSON cannot inject code."""
    bad = _minimal_spec()
    bad["edges"][0]["cond"] = "definitely_not_registered_anywhere"

    client = TestClient(create_app())
    r = client.post("/api/runs/from-spec", json={"spec": bad, "goal": "hi"})
    assert r.status_code == 422


async def test_from_spec_run_completes_with_stubbed_llm():
    """End-to-end: the spec actually runs to completion under the stub.

    Uses httpx.AsyncClient + ASGITransport so the FastAPI app and the
    background task share one event loop (the test's). TestClient would
    spin up its own loop per request and cancel the in-flight task.
    """
    import httpx

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/runs/from-spec",
            json={"spec": _minimal_spec(), "goal": "g"},
        )
        assert r.status_code == 201, r.text
        run_id = r.json()["run_id"]

        record = _routes._RUNS[run_id]
        assert record.task is not None
        await record.task

        final = (await client.get(f"/api/runs/{run_id}")).json()
        assert final["status"] == "done", final
        assert final["message_count"] >= 2


# ---------------------------------------------------------------------------
# /api/export-python
# ---------------------------------------------------------------------------

def test_export_python_renders_source():
    client = TestClient(create_app())
    r = client.post("/api/export-python", json={"spec": _minimal_spec()})
    assert r.status_code == 200
    src = r.json()["source"]
    assert "from fleet import Agent, Graph" in src
    assert "FleetLLM(" in src
    assert "graph = (" in src


def test_export_python_rejects_invalid_spec():
    bad = _minimal_spec()
    bad["entry"] = "ghost"
    client = TestClient(create_app())
    r = client.post("/api/export-python", json={"spec": bad})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /api/conditions and /api/tools
# ---------------------------------------------------------------------------

def test_conditions_route_lists_builtins():
    client = TestClient(create_app())
    r = client.get("/api/conditions")
    assert r.status_code == 200
    body = r.json()
    assert "always" in body["conditions"]
    assert "max_steps_not_hit" in body["conditions"]
    assert any(p.startswith("scratchpad_true") for p in body["parametric"])


def test_tools_route_lists_registered_tools():
    client = TestClient(create_app())
    r = client.get("/api/tools")
    assert r.status_code == 200
    tools = r.json()["tools"]
    # Built-ins from the registry.
    assert "web_search" in tools
    assert "read_file" in tools
