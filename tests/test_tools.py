"""Tests for the @tool decorator, registry, schema generation, and built-in tools."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── keep a clean registry snapshot so decorator tests don't bleed across tests ──
from fleet.tools.base import get_tool, to_anthropic, to_openai, tool


# ---------------------------------------------------------------------------
# @tool decorator and registry
# ---------------------------------------------------------------------------

def test_tool_decorator_registers():
    @tool
    async def _my_test_tool(query: str) -> str:
        """A test tool."""
        return query

    entry = get_tool("_my_test_tool")
    assert entry is not None
    assert entry["name"] == "_my_test_tool"
    assert entry["description"] == "A test tool."


def test_tool_sync_fn_is_wrapped_async():
    @tool
    def _sync_tool(x: int) -> int:
        """Sync tool."""
        return x + 1

    import asyncio
    entry = get_tool("_sync_tool")
    assert entry is not None
    assert asyncio.iscoroutinefunction(entry["fn"])


def test_tool_schema_basic_types():
    @tool
    async def _schema_tool(name: str, count: int, ratio: float, flag: bool) -> str:
        """Schema test."""
        return ""

    params = get_tool("_schema_tool")["parameters"]
    props = params["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["flag"]["type"] == "boolean"
    assert set(params["required"]) == {"name", "count", "ratio", "flag"}


def test_tool_schema_optional_param():
    @tool
    async def _opt_tool(query: str, max_results: int = 5) -> str:
        """Optional test."""
        return ""

    params = get_tool("_opt_tool")["parameters"]
    assert "query" in params["required"]
    assert "max_results" not in params["required"]
    assert params["properties"]["max_results"]["default"] == 5


def test_tool_schema_list_param():
    @tool
    async def _list_tool(items: list[str]) -> str:
        """List test."""
        return ""

    params = get_tool("_list_tool")["parameters"]
    prop = params["properties"]["items"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "string"


def test_to_anthropic_format():
    @tool
    async def _ant_tool(q: str) -> str:
        """Anthropic fmt."""
        return q

    fmt = to_anthropic("_ant_tool")
    assert fmt["name"] == "_ant_tool"
    assert fmt["description"] == "Anthropic fmt."
    assert "input_schema" in fmt
    assert fmt["input_schema"]["type"] == "object"


def test_to_openai_format():
    @tool
    async def _oai_tool(q: str) -> str:
        """OpenAI fmt."""
        return q

    fmt = to_openai("_oai_tool")
    assert fmt["type"] == "function"
    fn = fmt["function"]
    assert fn["name"] == "_oai_tool"
    assert fn["description"] == "OpenAI fmt."
    assert fn["parameters"]["type"] == "object"


def test_get_tool_unknown_returns_none():
    assert get_tool("__totally_unknown_xyz__") is None


# ---------------------------------------------------------------------------
# python_exec
# ---------------------------------------------------------------------------

async def test_python_exec_simple():
    from fleet.tools.code import python_exec
    result = await python_exec("print('hello fleet')")
    assert "hello fleet" in result


async def test_python_exec_captures_stderr():
    from fleet.tools.code import python_exec
    result = await python_exec("import sys; sys.stderr.write('err msg')")
    assert "err msg" in result


async def test_python_exec_timeout():
    from fleet.tools.code import python_exec
    result = await python_exec("import time; time.sleep(30)", timeout=1)
    assert "timed out" in result.lower()


async def test_python_exec_syntax_error():
    from fleet.tools.code import python_exec
    result = await python_exec("def bad syntax!!!")
    assert "STDERR" in result or "SyntaxError" in result


# ---------------------------------------------------------------------------
# file tools
# ---------------------------------------------------------------------------

async def test_write_and_read_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_WORKSPACE", str(tmp_path))
    from fleet.tools.files import read_file, write_file

    await write_file("hello.txt", "world")
    content = await read_file("hello.txt")
    assert content == "world"


async def test_write_creates_subdirs(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_WORKSPACE", str(tmp_path))
    from fleet.tools.files import read_file, write_file

    await write_file("sub/dir/file.txt", "nested")
    content = await read_file("sub/dir/file.txt")
    assert content == "nested"


async def test_list_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_WORKSPACE", str(tmp_path))
    from fleet.tools.files import list_dir, write_file

    await write_file("a.txt", "")
    await write_file("b.txt", "")
    listing = await list_dir(".")
    assert "a.txt" in listing
    assert "b.txt" in listing


async def test_list_dir_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_WORKSPACE", str(tmp_path))
    from fleet.tools.files import list_dir

    result = await list_dir(".")
    assert result == "(empty)"


async def test_file_path_escape_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_WORKSPACE", str(tmp_path))
    from fleet.tools.files import read_file

    with pytest.raises(ValueError, match="escape"):
        await read_file("../../etc/passwd")


async def test_file_path_absolute_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("FLEET_WORKSPACE", str(tmp_path))
    from fleet.tools.files import read_file

    # An absolute path outside workspace should be caught by realpath check
    with pytest.raises(ValueError, match="escape"):
        await read_file("/etc/passwd")


# ---------------------------------------------------------------------------
# web_search (mocked)
# ---------------------------------------------------------------------------

async def test_web_search_returns_json(monkeypatch):
    fake_results = [{"title": "T", "href": "http://x.com", "body": "Body"}]

    mock_ddgs_cls = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_ctx.text = MagicMock(return_value=fake_results)
    mock_ddgs_cls.return_value = mock_ctx

    with patch("fleet.tools.web.DDGS", mock_ddgs_cls):
        from fleet.tools.web import web_search
        result = await web_search("test query", max_results=1)

    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert parsed[0]["title"] == "T"


# ---------------------------------------------------------------------------
# web_fetch (mocked)
# ---------------------------------------------------------------------------

async def test_web_fetch_strips_html():
    html = b"<html><head><title>T</title></head><body><p>Hello world</p></body></html>"

    mock_resp = MagicMock()
    mock_resp.content = html
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("fleet.tools.web.httpx.AsyncClient", return_value=mock_client):
        from fleet.tools.web import web_fetch
        result = await web_fetch("http://example.com")

    assert "Hello world" in result
    assert "<p>" not in result


async def test_web_fetch_1mb_cap():
    """Content beyond 1 MB is silently truncated before stripping."""
    big = b"<html><body>" + b"x" * 2_000_000 + b"</body></html>"

    mock_resp = MagicMock()
    mock_resp.content = big
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("fleet.tools.web.httpx.AsyncClient", return_value=mock_client):
        from fleet.tools.web import web_fetch
        result = await web_fetch("http://example.com")

    # Result derived from at most 1 MB of input
    assert len(result) <= 1_000_000


# ---------------------------------------------------------------------------
# pretty_print
# ---------------------------------------------------------------------------

async def test_pretty_print_registry_lookup():
    # Importing fleet.tools triggers the side-effect import of `printing`,
    # which is what populates the registry. Verify both that the entry exists
    # and that its schema was derived correctly from the type hints.
    import fleet.tools  # noqa: F401

    entry = get_tool("pretty_print")
    assert entry is not None
    assert entry["name"] == "pretty_print"
    assert "Print a formatted message" in entry["description"]

    params = entry["parameters"]
    props = params["properties"]
    assert props["message"]["type"] == "string"
    assert props["prefix"]["type"] == "string"
    assert props["prefix"]["default"] == "[fleet]"
    assert props["upper"]["type"] == "boolean"
    assert props["upper"]["default"] is False
    assert params["required"] == ["message"]


async def test_pretty_print_executes_and_prints(capsys):
    from fleet.tools.printing import pretty_print

    result = await pretty_print("hello world")
    captured = capsys.readouterr()

    assert result == "[fleet] hello world"
    assert "[fleet] hello world" in captured.out


async def test_pretty_print_upper_and_custom_prefix(capsys):
    from fleet.tools.printing import pretty_print

    result = await pretty_print("ship it", prefix=">>>", upper=True)
    captured = capsys.readouterr()

    assert result == ">>> SHIP IT"
    assert ">>> SHIP IT" in captured.out


def test_pretty_print_provider_schemas():
    # The tool should be exposable through both Anthropic and OpenAI shapes.
    ant = to_anthropic("pretty_print")
    assert ant["name"] == "pretty_print"
    assert ant["input_schema"]["properties"]["message"]["type"] == "string"

    oai = to_openai("pretty_print")
    assert oai["type"] == "function"
    assert oai["function"]["name"] == "pretty_print"


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------

def test_skill_decorator_registers():
    from fleet.skills.registry import _SKILL_REGISTRY, skill

    @skill
    async def _test_skill_xyz(state):  # type: ignore[override]
        """A test skill."""
        return state

    assert "_test_skill_xyz" in _SKILL_REGISTRY
    assert _SKILL_REGISTRY["_test_skill_xyz"]["description"] == "A test skill."


