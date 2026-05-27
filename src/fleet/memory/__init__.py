"""fleet.memory — ReasoningBank: experience-augmented agent memory."""

from fleet.memory.bank import ReasoningBank
from fleet.memory.embedders import Embedder, get_embedder, register as register_embedder
from fleet.memory.item import MemoryItem
from fleet.memory.matts import matts_run
from fleet.memory.stores import InMemoryStore, MemoryStore

__all__ = [
    "MemoryItem",
    "ReasoningBank",
    "Embedder",
    "MemoryStore",
    "InMemoryStore",
    "get_embedder",
    "register_embedder",
    "matts_run",
]

try:
    from fleet.memory.stores import SQLiteVecStore  # noqa: F401

    if SQLiteVecStore is not None:
        __all__.append("SQLiteVecStore")
except ImportError:
    pass
