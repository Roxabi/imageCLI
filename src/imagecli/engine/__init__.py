"""Backward-compatible re-export shim — real code lives in the engine subpackage:
``base`` (ABC + preflight), ``helpers`` (helpers/errors), ``registry`` (factory)."""

from __future__ import annotations

from .base import ImageEngine, preflight_check
from .helpers import (
    MIN_FREE_RAM_GB,
    EngineCapabilities,
    InsufficientResourcesError,
    get_compute_capability,
    warn_ignored_params,
)
from .registry import (
    _get_registry,  # noqa: F401  # pyright: ignore[reportUnusedImport] — re-exported for existing callers (tests/test_engine.py)
    get_engine,
    list_engines,
)

__all__ = [
    "EngineCapabilities",
    "ImageEngine",
    "InsufficientResourcesError",
    "MIN_FREE_RAM_GB",
    "get_compute_capability",
    "get_engine",
    "list_engines",
    "preflight_check",
    "warn_ignored_params",
]
