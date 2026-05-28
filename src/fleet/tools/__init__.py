from fleet.tools.base import _TOOL_REGISTRY, get_tool, to_anthropic, to_openai, tool

# Eagerly import built-in tool modules so their @tool decorators register at
# `import fleet` time. Without this, users must import each submodule manually
# before passing tool names to Agent(...). web.py guards its optional
# duckduckgo-search dependency, so importing it without the [search] extra is safe.
from fleet.tools import code as _code  # noqa: F401
from fleet.tools import files as _files  # noqa: F401
from fleet.tools import web as _web  # noqa: F401

__all__ = ["tool", "get_tool", "to_anthropic", "to_openai", "_TOOL_REGISTRY"]
