"""Security tests for the spec-run path.

POST /api/runs/from-spec accepts an untrusted browser-supplied spec. These
tests prove that the validators close the obvious attack surfaces:

1. Conditions are a fixed registry — names are never eval'd.
2. Tools are a fixed registry — unknown names are rejected.
3. base_url is restricted to http/https and known-safe hosts; cloud
   metadata endpoints are blocked; localhost is allowed for local LLMs.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fleet.graphspec import AgentSpec, GraphSpec, validate_base_url
from fleet.server.app import create_app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _spec_with_edge_cond(cond: str | None) -> dict:
    return {
        "version": "0.3",
        "name": "sec",
        "nodes": [
            {
                "id": "a",
                "type": "agent",
                "agent": {
                    "name": "a", "provider": "anthropic", "model": "claude-sonnet-4-6",
                    "system": "", "tools": [], "memory_bank": False,
                },
            },
            {
                "id": "b",
                "type": "agent",
                "agent": {
                    "name": "b", "provider": "anthropic", "model": "claude-sonnet-4-6",
                    "system": "", "tools": [], "memory_bank": False,
                },
            },
        ],
        "edges": [{"src": "a", "dst": "b", "cond": cond}],
        "entry": "a",
        "exit": "b",
    }


def _spec_with_base_url(provider: str, url: str) -> dict:
    spec = _spec_with_edge_cond(None)
    spec["nodes"][0]["agent"]["provider"] = provider
    spec["nodes"][0]["agent"]["base_url"] = url
    spec["nodes"][1]["agent"]["provider"] = provider
    spec["nodes"][1]["agent"]["base_url"] = url
    return spec


# ===========================================================================
# 1. CONDITIONS — registry only, no eval
# ===========================================================================

@pytest.mark.parametrize("attack", [
    "__import__('os').system('rm -rf /')",
    "eval('1+1')",
    "exec('print(1)')",
    "always; print('pwned')",        # statement injection
    "always or True",                # boolean injection
    "lambda s: True",                # lambda injection
    "open('/etc/passwd').read()",    # file read
    "../../etc/passwd",              # path traversal
    "{} or always",                  # expression
    "no_such_condition_42",          # plain unknown name
])
def test_condition_injection_rejected_at_validation(attack):
    """A malicious-looking name is rejected as an unknown condition, never
    parsed or executed in any way."""
    bad = _spec_with_edge_cond(attack)
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(bad)
    msg = str(exc.value)
    assert "unknown" in msg.lower() and "condition" in msg.lower()


def test_condition_injection_rejected_at_api():
    """Same protection enforced at the HTTP boundary (422, not 500/200)."""
    client = TestClient(create_app())
    bad = _spec_with_edge_cond("__import__('os').system('rm -rf /')")
    r = client.post("/api/runs/from-spec", json={"spec": bad, "goal": "x"})
    assert r.status_code == 422
    assert "Invalid spec" in r.text


def test_condition_parametric_form_only_matches_registered_prefix():
    """Even 'prefix:arg' must match a registered prefix; arbitrary 'foo:bar' fails."""
    bad = _spec_with_edge_cond("rmrf:/etc/passwd")
    with pytest.raises(ValidationError):
        GraphSpec.model_validate(bad)


# ===========================================================================
# 2. TOOLS — registry only
# ===========================================================================

@pytest.mark.parametrize("attack", [
    "no_such_tool",
    "../web_search",
    "os.system",
    "__import__",
    "",
])
def test_unknown_tool_rejected_at_validation(attack):
    bad = _spec_with_edge_cond(None)
    bad["nodes"][0]["agent"]["tools"] = [attack]
    with pytest.raises(ValidationError) as exc:
        GraphSpec.model_validate(bad)
    assert "Unknown tool" in str(exc.value)


def test_unknown_tool_rejected_at_api():
    client = TestClient(create_app())
    bad = _spec_with_edge_cond(None)
    bad["nodes"][0]["agent"]["tools"] = ["__import__"]
    r = client.post("/api/runs/from-spec", json={"spec": bad, "goal": "x"})
    assert r.status_code == 422


# ===========================================================================
# 3. base_url SSRF guard
# ===========================================================================

# ── Blocked: non-http(s) schemes ────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://example.com/_GET%20/secret",
    "ftp://example.com/",
    "dict://localhost:11211/stats",
    "ldap://x.example.com/",
    "data:text/plain,hi",
    "javascript:alert(1)",
])
def test_non_http_schemes_rejected(url):
    with pytest.raises(ValueError) as exc:
        validate_base_url(url)
    assert "http" in str(exc.value)


# ── Blocked: cloud metadata endpoints ───────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",            # AWS
    "https://169.254.169.254/latest/api/token",            # AWS IMDSv2
    "http://169.254.170.2/v2/credentials",                 # ECS task
    "http://metadata.google.internal/computeMetadata/v1/", # GCP
    "http://metadata.goog/computeMetadata/v1/",            # GCP alt
    "http://100.100.100.200/",                             # Alibaba
    "http://169.254.169.253/",                             # Azure IMDS
])
def test_cloud_metadata_endpoints_rejected(url):
    with pytest.raises(ValueError) as exc:
        validate_base_url(url)
    assert "metadata" in str(exc.value).lower()


# ── Allowed: localhost + RFC1918 (legit local LLMs) ─────────────────────────

@pytest.mark.parametrize("url", [
    "http://localhost:11434",
    "http://localhost:11434/v1",
    "http://127.0.0.1:8000/v1",
    "https://localhost:8443/v1",
    "http://0.0.0.0:8000",
    "http://[::1]:11434/v1",
    "http://192.168.1.50:8000/v1",
    "http://10.0.0.5:8000",
    # And public hosted providers:
    "https://api.deepseek.com",
    "https://api.together.xyz/v1",
])
def test_safe_base_urls_accepted(url):
    # No exception.
    validate_base_url(url)


# ── End-to-end: SSRF guard enforced via AgentSpec ───────────────────────────

def test_agentspec_rejects_file_url():
    with pytest.raises(ValidationError) as exc:
        AgentSpec(
            name="x", provider="custom", model="m",
            base_url="file:///etc/passwd",
        )
    assert "http" in str(exc.value).lower()


def test_agentspec_rejects_aws_metadata():
    with pytest.raises(ValidationError) as exc:
        AgentSpec(
            name="x", provider="custom", model="m",
            base_url="http://169.254.169.254/latest/meta-data/",
        )
    assert "metadata" in str(exc.value).lower()


def test_agentspec_accepts_localhost():
    spec = AgentSpec(
        name="x", provider="custom", model="m",
        base_url="http://localhost:11434/v1",
    )
    assert spec.base_url == "http://localhost:11434/v1"


# ── End-to-end: SSRF guard enforced at the /api/runs/from-spec boundary ─────

def test_api_rejects_file_base_url():
    client = TestClient(create_app())
    spec = _spec_with_base_url("custom", "file:///etc/passwd")
    r = client.post("/api/runs/from-spec", json={"spec": spec, "goal": "x"})
    assert r.status_code == 422


def test_api_rejects_aws_metadata_base_url():
    client = TestClient(create_app())
    spec = _spec_with_base_url("custom", "http://169.254.169.254/latest/")
    r = client.post("/api/runs/from-spec", json={"spec": spec, "goal": "x"})
    assert r.status_code == 422
    assert "metadata" in r.text.lower()


def test_api_rejects_gcp_metadata_base_url():
    client = TestClient(create_app())
    spec = _spec_with_base_url("custom", "http://metadata.google.internal/computeMetadata/v1/")
    r = client.post("/api/runs/from-spec", json={"spec": spec, "goal": "x"})
    assert r.status_code == 422


def test_api_accepts_localhost_base_url(monkeypatch):
    """Localhost is allowed — local vLLM/Ollama is a first-class use case."""
    # Stub FleetLLM.complete so the loader build doesn't try a real network call.
    from fleet.providers import client as _c
    from fleet.core.messages import AgentMessage

    async def _fake(self, messages, tools=None):  # noqa: ARG001
        return AgentMessage(role="assistant", content="ok")

    monkeypatch.setattr(_c.FleetLLM, "complete", _fake)
    monkeypatch.setenv("CUSTOM_API_KEY", "dummy")

    client = TestClient(create_app())
    spec = _spec_with_base_url("custom", "http://localhost:11434/v1")
    r = client.post("/api/runs/from-spec", json={"spec": spec, "goal": "x"})
    # Must NOT be rejected by the SSRF guard (validation passes → run starts).
    assert r.status_code == 201, r.text


def test_api_accepts_127_base_url(monkeypatch):
    from fleet.providers import client as _c
    from fleet.core.messages import AgentMessage

    async def _fake(self, messages, tools=None):  # noqa: ARG001
        return AgentMessage(role="assistant", content="ok")

    monkeypatch.setattr(_c.FleetLLM, "complete", _fake)
    monkeypatch.setenv("CUSTOM_API_KEY", "dummy")

    client = TestClient(create_app())
    spec = _spec_with_base_url("custom", "http://127.0.0.1:8000/v1")
    r = client.post("/api/runs/from-spec", json={"spec": spec, "goal": "x"})
    assert r.status_code == 201, r.text
