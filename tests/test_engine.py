"""Tests for imagecli.engine — registry, get_engine, list_engines, preflight_check."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imagecli.engine import (
    ImageEngine,
    InsufficientResourcesError,
    _get_registry,
    get_engine,
    list_engines,
    preflight_check,
)


def test_registry_has_all_engines():
    # Act
    registry = _get_registry()

    # Assert
    assert "flux2-klein" in registry
    assert "flux1-dev" in registry
    assert "flux1-schnell" in registry
    assert "sd35" in registry


def test_get_engine_valid():
    # Act
    engine = get_engine("flux2-klein")

    # Assert
    assert isinstance(engine, ImageEngine)
    assert engine.name == "flux2-klein"


def test_get_engine_invalid():
    # Act / Assert
    with pytest.raises(ValueError, match="nonexistent"):
        get_engine("nonexistent")


def test_list_engines():
    # Act
    engines = list_engines()

    # Assert: returns a non-empty list
    assert isinstance(engines, list)
    assert len(engines) > 0

    # Assert: each entry has the required keys
    required_keys = {"name", "description", "model_id", "vram_gb"}
    for entry in engines:
        for key in required_keys:
            assert key in entry, f"Engine entry missing key {key!r}: {entry}"


def test_preflight_no_cuda():
    # Arrange: simulate no CUDA GPU available.
    # torch is imported locally inside preflight_check, so we patch it in sys.modules.
    engine = get_engine("flux2-klein")

    with patch("torch.cuda.is_available", return_value=False):
        # Act / Assert
        with pytest.raises(InsufficientResourcesError, match="No CUDA GPU detected"):
            preflight_check(engine)
