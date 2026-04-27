"""Tests for engine subpackage split — import isolation, __all__ completeness, no circulars."""

from __future__ import annotations

import importlib
import types


# ── 2a — Submodule imports ───────────────────────────────────────────────────


def test_engine_base_imports():
    mod = importlib.import_module("imagecli.engine.base")
    assert hasattr(mod, "ImageEngine")
    assert hasattr(mod, "preflight_check")


def test_engine_helpers_imports():
    mod = importlib.import_module("imagecli.engine.helpers")
    assert hasattr(mod, "EngineCapabilities")
    assert hasattr(mod, "InsufficientResourcesError")
    assert hasattr(mod, "MIN_FREE_RAM_GB")
    assert hasattr(mod, "get_compute_capability")
    assert hasattr(mod, "warn_ignored_params")


def test_engine_registry_imports():
    mod = importlib.import_module("imagecli.engine.registry")
    assert hasattr(mod, "get_engine")
    assert hasattr(mod, "list_engines")


def test_engine_shim_imports():
    mod = importlib.import_module("imagecli.engine")
    assert hasattr(mod, "ImageEngine")
    assert hasattr(mod, "preflight_check")
    assert hasattr(mod, "EngineCapabilities")
    assert hasattr(mod, "InsufficientResourcesError")
    assert hasattr(mod, "MIN_FREE_RAM_GB")
    assert hasattr(mod, "get_compute_capability")
    assert hasattr(mod, "warn_ignored_params")
    assert hasattr(mod, "get_engine")
    assert hasattr(mod, "list_engines")


# ── 2b — __all__ completeness on the shim ───────────────────────────────────


def test_shim_all_names_exist():
    mod = importlib.import_module("imagecli.engine")
    assert hasattr(mod, "__all__"), "imagecli.engine must define __all__"
    for name in mod.__all__:
        assert hasattr(mod, name), f"imagecli.engine.__all__ declares {name!r} but it is missing"


# ── 2c — No circular imports ─────────────────────────────────────────────────


def test_no_circular_engine_base():
    mod = importlib.import_module("imagecli.engine.base")
    assert isinstance(mod, types.ModuleType)


def test_no_circular_engine_helpers():
    mod = importlib.import_module("imagecli.engine.helpers")
    assert isinstance(mod, types.ModuleType)


def test_no_circular_engine_registry():
    mod = importlib.import_module("imagecli.engine.registry")
    assert isinstance(mod, types.ModuleType)


def test_no_circular_engine_shim():
    mod = importlib.import_module("imagecli.engine")
    assert isinstance(mod, types.ModuleType)
