from __future__ import annotations

import importlib.util
import inspect
import os
from typing import Any, Callable

from fleet.core.state import GraphState

_SKILL_REGISTRY: dict[str, dict[str, Any]] = {}


def skill(fn: Callable[[GraphState], Any]) -> Callable[[GraphState], Any]:
    """Decorator: registers an async function as a named fleet skill."""
    name = fn.__name__
    _SKILL_REGISTRY[name] = {
        "fn": fn,
        "name": name,
        "description": inspect.getdoc(fn) or "",
    }
    return fn


def get_skill(name: str) -> dict[str, Any] | None:
    return _SKILL_REGISTRY.get(name)


def discover_skills(skills_dir: str) -> None:
    """Import every *.py file in *skills_dir* so their @skill decorators fire."""
    for fname in sorted(os.listdir(skills_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        path = os.path.join(skills_dir, fname)
        mod_name = fname[:-3]
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
