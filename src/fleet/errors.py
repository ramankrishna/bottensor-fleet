"""Fleet-level exception hierarchy."""
from __future__ import annotations


class FleetError(Exception):
    """Base class for all fleet errors."""


class ProviderError(FleetError):
    """Raised when an LLM provider call fails after all retries.

    Wraps the underlying polyrt / SDK error with a clean one-line message.
    """
