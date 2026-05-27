"""Tests for fleet.memory.embedders.minilm.MiniLMEmbedder."""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from fleet.memory.embedders import Embedder, get_embedder  # noqa: E402
from fleet.memory.embedders.minilm import DEFAULT_DIM, MiniLMEmbedder  # noqa: E402


def test_protocol_conformance() -> None:
    embedder = MiniLMEmbedder()
    assert isinstance(embedder, Embedder)
    assert embedder.dim == DEFAULT_DIM


def test_registry_returns_minilm() -> None:
    embedder = get_embedder("minilm")
    assert isinstance(embedder, MiniLMEmbedder)


def test_embed_shape_and_distinct() -> None:
    embedder = MiniLMEmbedder()
    texts = [
        "ripgrep is faster than grep on large repos",
        "use uv to manage python dependencies",
        "the sky is blue today",
    ]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == DEFAULT_DIM
        assert all(isinstance(v, float) for v in vec)
    # Semantically distinct texts should not collapse to identical vectors.
    assert vectors[0] != vectors[1]
    assert vectors[0] != vectors[2]
