from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fleet.core.state import GraphState


class CheckpointBackend(Protocol):
    async def save(self, run_id: str, state: GraphState) -> None: ...
    async def load(self, run_id: str) -> GraphState | None: ...
    async def list_runs(self) -> list[str]: ...


class SQLiteCheckpoint:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.path.expanduser("~/.fleet/checkpoints.db")

    async def save(self, run_id: str, state: GraphState) -> None:
        import aiosqlite

        db_dir = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(db_dir, exist_ok=True)

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints "
                "(run_id TEXT PRIMARY KEY, state_json TEXT, saved_at TEXT)"
            )
            await db.execute(
                "INSERT OR REPLACE INTO checkpoints (run_id, state_json, saved_at) "
                "VALUES (?, ?, ?)",
                (run_id, state.model_dump_json(), datetime.utcnow().isoformat()),
            )
            await db.commit()

    async def load(self, run_id: str) -> GraphState | None:
        import aiosqlite

        from fleet.core.state import GraphState

        if not os.path.exists(self.path):
            return None

        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT state_json FROM checkpoints WHERE run_id = ?", (run_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return GraphState.model_validate_json(row[0])

    async def list_runs(self) -> list[str]:
        import aiosqlite

        if not os.path.exists(self.path):
            return []

        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT run_id FROM checkpoints ORDER BY saved_at"
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]


class RedisCheckpoint:
    def __init__(self, url: str | None = None) -> None:
        self.url = url or os.environ.get("FLEET_REDIS_URL", "redis://localhost:6379")

    async def save(self, run_id: str, state: GraphState) -> None:
        import redis.asyncio as aioredis

        async with aioredis.from_url(self.url) as client:
            await client.set(f"fleet:checkpoint:{run_id}", state.model_dump_json())

    async def load(self, run_id: str) -> GraphState | None:
        import redis.asyncio as aioredis

        from fleet.core.state import GraphState

        async with aioredis.from_url(self.url) as client:
            data = await client.get(f"fleet:checkpoint:{run_id}")
            if data is None:
                return None
            return GraphState.model_validate_json(data)

    async def list_runs(self) -> list[str]:
        import redis.asyncio as aioredis

        prefix = "fleet:checkpoint:"
        async with aioredis.from_url(self.url) as client:
            keys = await client.keys(f"{prefix}*")
            return [
                k.decode().removeprefix(prefix) if isinstance(k, bytes) else k.removeprefix(prefix)
                for k in keys
            ]
