from __future__ import annotations

import sys

from fleet.tools.base import tool


@tool
def pretty_print(message: str, prefix: str = "[fleet]", upper: bool = False) -> str:
    """Print a formatted message to stdout and return the rendered string.

    The output is `"<prefix> <message>"`, optionally upper-cased.
    """
    body = message.upper() if upper else message
    line = f"{prefix} {body}"
    print(line, file=sys.stdout, flush=True)
    return line
