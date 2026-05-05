from __future__ import annotations

import asyncio
import json
from html.parser import HTMLParser

import httpx
from duckduckgo_search import DDGS

from fleet.tools.base import tool


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "head"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _strip_html(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    return parser.get_text()


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return JSON results."""
    def _sync_search() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    results = await asyncio.to_thread(_sync_search)
    return json.dumps(results, indent=2)


@tool
async def web_fetch(url: str) -> str:
    """Fetch a URL and return plain text (1 MB cap, HTML stripped)."""
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        raw = resp.content[:1_000_000]
    return _strip_html(raw.decode(errors="replace"))
