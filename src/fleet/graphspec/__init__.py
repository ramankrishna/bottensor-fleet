"""JSON graph specification for Fleet.

Define a multi-agent graph declaratively as a small JSON document, load it
into a runnable :class:`fleet.CompiledGraph`, and (optionally) export it
back to a hand-readable Python file.

Public surface:
    GraphSpec, NodeSpec, AgentSpec, EdgeSpec, Position
    load_graph_spec
    register_condition, get_condition
    spec_to_python
"""

from fleet.graphspec.spec import (
    AgentSpec,
    EdgeSpec,
    GraphSpec,
    NodeSpec,
    Position,
    SUPPORTED_PROVIDERS,
)
from fleet.graphspec.conditions import (
    get_condition,
    register_condition,
)
from fleet.graphspec.loader import load_graph_spec

__all__ = [
    "AgentSpec",
    "EdgeSpec",
    "GraphSpec",
    "NodeSpec",
    "Position",
    "SUPPORTED_PROVIDERS",
    "get_condition",
    "register_condition",
    "load_graph_spec",
]
