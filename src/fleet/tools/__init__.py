from fleet.tools.base import _TOOL_REGISTRY, get_tool, to_anthropic, to_openai, tool

# Eagerly import built-in tool modules so the @tool decorator runs and the
# registry is populated as a side effect of `import fleet`.
# Each import is guarded so a missing optional dep (e.g. `ddgs`/`duckduckgo_search`
# for web_search) doesn't break the whole package.
from fleet.tools import files as _files  # noqa: F401
from fleet.tools import code as _code  # noqa: F401
from fleet.tools import printing as _printing  # noqa: F401

try:
    from fleet.tools import web as _web  # noqa: F401
except Exception:
    # web module imports httpx (required) but the file is fine to register
    # even if duckduckgo search is missing — web_fetch is still useful.
    pass

__all__ = ["tool", "get_tool", "to_anthropic", "to_openai", "_TOOL_REGISTRY"]
