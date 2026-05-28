"""MLX embedder — stub (coming v0.3).

Apple-Silicon-native embedding path that will use `mlx-embeddings` once the
real bridge lands. The class is intentionally not registered yet so that
``get_embedder("mlx")`` raises ``KeyError`` instead of returning an object
that fails on first ``embed()`` call.
"""

from __future__ import annotations


class MLXEmbedder:
    """Stub MLX embedder (coming v0.3). Do not use yet — raises NotImplementedError."""

    @property
    def dim(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("MLX embedder is a stub — coming in v0.3")
