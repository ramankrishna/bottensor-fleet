from __future__ import annotations

import asyncio
import os
import sys

from fleet.tools.base import tool

_DEFAULT_WORKSPACE = os.path.expanduser("~/.fleet/workspace")


def _workspace() -> str:
    return os.environ.get("FLEET_WORKSPACE", _DEFAULT_WORKSPACE)


@tool
async def python_exec(code: str, timeout: int = 10) -> str:
    """Execute Python code in an isolated subprocess and return stdout + stderr."""
    ws = _workspace()
    os.makedirs(ws, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=ws,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=float(timeout))
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.communicate()
        except Exception:
            pass
        return f"Error: execution timed out after {timeout}s"

    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace").strip()
    if err:
        out = out + f"\nSTDERR:\n{err}"
    return out
