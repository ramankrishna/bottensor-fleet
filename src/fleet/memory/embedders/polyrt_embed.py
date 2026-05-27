"""Polyrt-hosted embedder — stub. Wires to polyrt's embeddings API later."""

from __future__ import annotations


class PolyRTEmbedder:
    @property
    def dim(self) -> int:
        return 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("polyrt-hosted embedder will land in a later v0.2 phase")
