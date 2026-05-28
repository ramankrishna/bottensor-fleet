"""Embedder protocol + registry for the ReasoningBank.

Embedders turn signature_text() into a fixed-dimension vector that backs
similarity search. Implementations are registered by name so callers can
swap MiniLM, MLX, or polyrt-hosted embeddings without code changes.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns strings into fixed-length vectors."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings. Returns one vector per text."""
        ...


EmbedderFactory = Callable[..., Embedder]
_REGISTRY: dict[str, EmbedderFactory] = {}


def register(name: str, factory: EmbedderFactory) -> None:
    _REGISTRY[name] = factory


def get_embedder(name: str, **kwargs: object) -> Embedder:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown embedder {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


# Built-in registrations — factories are lazy so optional deps aren't required at import.
def _minilm_factory(**kwargs: object) -> Embedder:
    from fleet.memory.embedders.minilm import MiniLMEmbedder
    return MiniLMEmbedder(**kwargs)  # type: ignore[arg-type]


def _mlx_factory(**kwargs: object) -> Embedder:
    from fleet.memory.embedders.mlx_embed import MLXEmbedder
    return MLXEmbedder(**kwargs)  # type: ignore[arg-type]


def _polyrt_factory(**kwargs: object) -> Embedder:
    from fleet.memory.embedders.polyrt_embed import PolyRTEmbedder
    return PolyRTEmbedder(**kwargs)  # type: ignore[arg-type]


register("minilm", _minilm_factory)
register("mlx", _mlx_factory)
register("polyrt", _polyrt_factory)
