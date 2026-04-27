"""Backward-compatible re-export shim — real code lives in the engine subpackage:
``base`` (ABC), ``helpers`` (helpers/errors/preflight), ``registry`` (factory)."""

from __future__ import annotations

from .base import ImageEngine
from .helpers import (
    MIN_FREE_RAM_GB,
    EngineCapabilities,
    InsufficientResourcesError,
    get_compute_capability,
    preflight_check,
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
