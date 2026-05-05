from __future__ import annotations

import asyncio
import functools
import inspect
import typing
from typing import Any, Callable

_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def tool(fn: Callable) -> Callable:
    """Decorator: registers a callable as a fleet tool with auto-generated JSON Schema."""
    name = fn.__name__
    description = inspect.getdoc(fn) or ""
    parameters = _build_schema(fn)

    # Wrap sync callables so the Agent can always `await` them.
    if asyncio.iscoroutinefunction(fn):
        async_fn = fn
    else:
        @functools.wraps(fn)
        async def async_fn(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            return await asyncio.to_thread(fn, *args, **kwargs)

    _TOOL_REGISTRY[name] = {
        "fn": async_fn,
        "name": name,
        "description": description,
        "parameters": parameters,
    }
    return async_fn


def get_tool(name: str) -> dict[str, Any] | None:
    return _TOOL_REGISTRY.get(name)


def to_anthropic(name: str) -> dict[str, Any]:
    """Return the Anthropic tool-use schema for a registered tool."""
    t = _TOOL_REGISTRY[name]
    return {
        "name": t["name"],
        "description": t["description"],
        "input_schema": t["parameters"],
    }


def to_openai(name: str) -> dict[str, Any]:
    """Return the OpenAI function-calling schema for a registered tool."""
    t = _TOOL_REGISTRY[name]
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        },
    }


# ---------------------------------------------------------------------------
# JSON Schema generation
# ---------------------------------------------------------------------------

def _build_schema(fn: Callable) -> dict[str, Any]:
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("return", "self", "cls"):
            continue
        hint = hints.get(param_name, Any)
        prop = _hint_to_schema(hint)
        if param.default is not inspect.Parameter.empty:
            prop = {**prop, "default": param.default}
        else:
            required.append(param_name)
        properties[param_name] = prop

    return {"type": "object", "properties": properties, "required": required}


def _hint_to_schema(hint: Any) -> dict[str, Any]:
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if hint is str:
        return {"type": "string"}
    if hint is int:
        return {"type": "integer"}
    if hint is float:
        return {"type": "number"}
    if hint is bool:
        return {"type": "boolean"}
    if origin is list or hint is list:
        item_schema = _hint_to_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    if origin is dict or hint is dict:
        return {"type": "object"}

    # Optional[X] → Union[X, None]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _hint_to_schema(non_none[0])

    return {"type": "string"}  # fallback
