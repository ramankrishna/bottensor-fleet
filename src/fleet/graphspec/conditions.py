"""Named condition registry for JSON graph edges.

JSON graphs cannot reference arbitrary Python predicates. Instead, edges
name a condition that has been pre-registered via :func:`register_condition`.
This keeps the JSON format declarative and safe — no ``eval``, no imports
of arbitrary modules at load time.

Built-in conditions:
    ``always``                     — always True (equivalent to no condition)
    ``scratchpad_true:<key>``      — True when ``state.scratchpad[key]`` is truthy
    ``max_steps_not_hit``          — True while ``state.metadata['terminated_by']``
                                     has not been set to ``'max_steps'``

Custom conditions may be registered at runtime — they apply per-process and
are NOT serialised into the JSON spec.
"""
from __future__ import annotations

from typing import Callable

from fleet.core.state import GraphState

ConditionFn = Callable[[GraphState], bool]

_CONDITION_REGISTRY: dict[str, ConditionFn] = {}
_PARAMETRIC_REGISTRY: dict[str, Callable[[str], ConditionFn]] = {}


def register_condition(name: str, fn: ConditionFn) -> None:
    """Register a named condition usable from a JSON graph spec."""
    if not name:
        raise ValueError("Condition name must be a non-empty string.")
    if ":" in name:
        raise ValueError(
            f"Condition name '{name}' cannot contain ':'. "
            "Use register_parametric_condition for prefix-style names."
        )
    _CONDITION_REGISTRY[name] = fn


def register_parametric_condition(
    prefix: str, factory: Callable[[str], ConditionFn]
) -> None:
    """Register a prefix-style condition: name 'prefix:<arg>' calls ``factory(arg)``."""
    if not prefix:
        raise ValueError("Condition prefix must be a non-empty string.")
    _PARAMETRIC_REGISTRY[prefix] = factory


def get_condition(name: str) -> ConditionFn | None:
    """Return the predicate for ``name``, or None if not registered."""
    if name in _CONDITION_REGISTRY:
        return _CONDITION_REGISTRY[name]
    if ":" in name:
        prefix, arg = name.split(":", 1)
        factory = _PARAMETRIC_REGISTRY.get(prefix)
        if factory is not None:
            return factory(arg)
    return None


# ---------------------------------------------------------------------------
# built-ins
# ---------------------------------------------------------------------------

def _always(_state: GraphState) -> bool:
    return True


def _max_steps_not_hit(state: GraphState) -> bool:
    return state.metadata.get("terminated_by") != "max_steps"


def _scratchpad_true(key: str) -> ConditionFn:
    def _pred(state: GraphState) -> bool:
        return bool(state.scratchpad.get(key))

    _pred.__name__ = f"scratchpad_true_{key}"
    return _pred


register_condition("always", _always)
register_condition("max_steps_not_hit", _max_steps_not_hit)
register_parametric_condition("scratchpad_true", _scratchpad_true)
