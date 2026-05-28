"""MLX embedder — stub. Apple-Silicon-native path coming in a later phase."""

from __future__ import annotations


class MLXEmbedder:
    @property
    def dim(self) -> int:
        return 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("MLX embedder will land in a later v0.2 phase")
