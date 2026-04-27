"""Backward-compatible re-export shim — real code is split across
``engine_base`` (ABC + preflight), ``engine_helpers`` (helpers/errors),
and ``engine_registry`` (factory)."""

from __future__ import annotations

from imagecli.engine_base import ImageEngine
from imagecli.engine_helpers import (
    preflight_check,
    MIN_FREE_RAM_GB,
    EngineCapabilities,
    InsufficientResourcesError,
    get_compute_capability,
    warn_ignored_params,
)
from imagecli.engine_registry import (
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
