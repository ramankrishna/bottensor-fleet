"""Construction-only tests for DeepSeek + custom OpenAI-compatible providers.

No live API calls — these check spec validation, FleetLLM construction, API
key resolution, and the /api/providers route surface.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fleet.errors import ProviderError
from fleet.graphspec import AgentSpec, SUPPORTED_PROVIDERS
from fleet.providers.client import FleetLLM


# ---------------------------------------------------------------------------
# AgentSpec — provider list & base_url rules
# ---------------------------------------------------------------------------

def test_deepseek_in_supported_providers():
    assert "deepseek" in SUPPORTED_PROVIDERS
    assert "custom" in SUPPORTED_PROVIDERS


def test_agentspec_deepseek_does_not_require_base_url():
    spec = AgentSpec(
        name="r",
        provider="deepseek",
        model="deepseek-chat",
        system="be brief",
    )
    assert spec.base_url is None


def test_agentspec_deepseek_accepts_optional_base_url():
    spec = AgentSpec(
        name="r",
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://custom-deepseek.example/v1",
    )
    assert spec.base_url == "https://custom-deepseek.example/v1"


def test_agentspec_custom_without_base_url_raises():
    with pytest.raises(ValidationError) as exc:
        AgentSpec(
            name="local",
            provider="custom",
            model="llama-3-8b-instruct",
        )
    assert "base_url" in str(exc.value)


def test_agentspec_custom_with_base_url_accepted():
    spec = AgentSpec(
        name="local",
        provider="custom",
        model="llama-3-8b-instruct",
        base_url="http://localhost:11434/v1",
    )
    assert spec.base_url == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# FleetLLM construction — no network calls
# ---------------------------------------------------------------------------

def test_fleetllm_deepseek_default_base_url():
    llm = FleetLLM("deepseek", "deepseek-chat")
    assert llm.backend == "deepseek"
    assert llm._extra["base_url"] == "https://api.deepseek.com"


def test_fleetllm_deepseek_user_base_url_wins():
    llm = FleetLLM("deepseek", "deepseek-chat", base_url="https://proxy.example/v1")
    assert llm._extra["base_url"] == "https://proxy.example/v1"


def test_fleetllm_custom_requires_base_url():
    with pytest.raises(ProviderError) as exc:
        FleetLLM("custom", "llama-3")
    assert "base_url" in str(exc.value)


def test_fleetllm_custom_constructs_with_base_url():
    llm = FleetLLM("custom", "llama-3", base_url="http://localhost:8000/v1")
    assert llm.backend == "custom"
    assert llm._extra["base_url"] == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def test_deepseek_uses_deepseek_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-123")
    llm = FleetLLM("deepseek", "deepseek-chat")
    assert llm._resolve_api_key() == "ds-test-123"


def test_deepseek_falls_back_to_openai_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback")
    llm = FleetLLM("deepseek", "deepseek-chat")
    assert llm._resolve_api_key() == "sk-fallback"


def test_custom_uses_custom_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-abc")
    llm = FleetLLM("custom", "llama-3", base_url="http://localhost:11434/v1")
    assert llm._resolve_api_key() == "custom-abc"


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "wrong")
    llm = FleetLLM("deepseek", "deepseek-chat", api_key="explicit")
    assert llm._resolve_api_key() == "explicit"


# ---------------------------------------------------------------------------
# /api/providers exposes the new backends
# ---------------------------------------------------------------------------

def test_providers_route_lists_deepseek_and_custom():
    from fleet.server.app import create_app

    client = TestClient(create_app())
    r = client.get("/api/providers")
    assert r.status_code == 200
    body = r.json()
    assert "deepseek" in body["providers"]
    assert "custom" in body["providers"]
    assert "openai" in body["providers"]
    assert "anthropic" in body["providers"]

    # Per-provider details carry the UI hints
    details_by_name = {d["name"]: d for d in body["details"]}
    assert details_by_name["custom"]["requires_base_url"] is True
    assert details_by_name["custom"]["default_base_url"] is None
    assert details_by_name["deepseek"]["requires_base_url"] is False
    assert details_by_name["deepseek"]["default_base_url"] == "https://api.deepseek.com"
    assert details_by_name["anthropic"]["requires_base_url"] is False
