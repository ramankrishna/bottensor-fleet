"""Export/import round-trip tests for ReasoningBank."""

from __future__ import annotations

from pathlib import Path

from fleet.memory import ReasoningBank
from fleet.memory.stores.inmemory import InMemoryStore


class _StaticEmbedder:
    DIM = 4

    @property
    def dim(self) -> int:
        return self.DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            for i, ch in enumerate(text.lower()):
                vec[(ord(ch) + i) % self.DIM] += 1.0
            n = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / n for v in vec])
        return out


def _bank() -> ReasoningBank:
    return ReasoningBank(store=InMemoryStore(), embedder=_StaticEmbedder())


async def test_export_then_import_round_trip(tmp_path: Path) -> None:
    src = _bank()
    a = await src.add_manual(title="A", description="da", content="ca")
    b = await src.add_manual(title="B", description="db", content="cb")

    out = tmp_path / "memories.jsonl"
    await src.export(out)
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    dst = _bank()
    n = await dst.import_(out)
    assert n == 2
    imported_ids = {it.id for it in await dst.list()}
    assert imported_ids == {a.id, b.id}

    # Field preserved across the round trip.
    imported = {it.id: it for it in await dst.list()}
    assert imported[a.id].title == "A"
    assert imported[a.id].content == "ca"


async def test_import_idempotent_on_duplicate_ids(tmp_path: Path) -> None:
    bank = _bank()
    await bank.add_manual(title="A", description="d", content="c")
    out = tmp_path / "m.jsonl"
    await bank.export(out)

    # Re-importing the same file does not raise; duplicates are upserted.
    n = await bank.import_(out)
    assert n == 1
    assert len(await bank.list()) == 1


async def test_export_creates_parent_dir(tmp_path: Path) -> None:
    bank = _bank()
    await bank.add_manual(title="x", description="d", content="c")
    nested = tmp_path / "a" / "b" / "out.jsonl"
    await bank.export(nested)
    assert nested.exists()
