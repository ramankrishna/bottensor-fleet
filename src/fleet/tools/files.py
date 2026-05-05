from __future__ import annotations

import os

from fleet.tools.base import tool

_DEFAULT_WORKSPACE = os.path.expanduser("~/.fleet/workspace")


def _workspace() -> str:
    return os.environ.get("FLEET_WORKSPACE", _DEFAULT_WORKSPACE)


def _safe_path(rel: str) -> str:
    ws = os.path.realpath(_workspace())
    target = os.path.realpath(os.path.join(ws, rel))
    if target != ws and not target.startswith(ws + os.sep):
        raise ValueError(f"Path escape detected: {rel!r} resolves outside workspace")
    return target


@tool
async def read_file(path: str) -> str:
    """Read a file from the fleet workspace and return its contents."""
    safe = _safe_path(path)
    with open(safe, encoding="utf-8") as fh:
        return fh.read()


@tool
async def write_file(path: str, content: str) -> str:
    """Write content to a file in the fleet workspace. Creates parent dirs as needed."""
    safe = _safe_path(path)
    os.makedirs(os.path.dirname(safe), exist_ok=True)
    with open(safe, "w", encoding="utf-8") as fh:
        fh.write(content)
    return f"Wrote {len(content)} bytes to {path}"


@tool
async def list_dir(path: str = ".") -> str:
    """List directory contents inside the fleet workspace."""
    safe = _safe_path(path)
    if not os.path.isdir(safe):
        return f"Error: {path!r} is not a directory"
    entries = sorted(os.listdir(safe))
    return "\n".join(entries) if entries else "(empty)"
