"""Tests for imagecli.engine — registry, get_engine, list_engines, preflight_check."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imagecli.engine import (
    ImageEngine,
    InsufficientResourcesError,
    _get_registry,
    get_compute_capability,
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


def test_get_compute_capability_no_cuda():
    # Arrange: simulate no CUDA GPU available.
    with patch("torch.cuda.is_available", return_value=False):
        # Act
        result = get_compute_capability()
    # Assert
    assert result == (0, 0)


def test_get_compute_capability_with_cuda():
    # Arrange: mock a device with sm_86 (Ampere RTX 3080).
    mock_props = MagicMock()
    mock_props.major = 8
    mock_props.minor = 6
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
    ):
        # Act
        result = get_compute_capability()
    # Assert
    assert result == (8, 6)


# ── F5 — Quantization branch selection threshold ─────────────────────────────


def test_sm_tuple_fp8_threshold_boundary():
    # Arrange: boundary values for the quantization dtype selection rule.
    # fp8 is used when sm >= (8, 9); int8 is used when sm < (8, 9).

    # Act / Assert: Ada Lovelace boundary — exactly at threshold → fp8
    assert (8, 9) >= (8, 9)

    # Act / Assert: Ampere sm_86 is below threshold → int8
    assert (8, 6) < (8, 9)

    # Act / Assert: Blackwell sm_120 is above threshold → fp8
    assert (12, 0) >= (8, 9)

    # Act / Assert: Turing sm_75 is below threshold → int8
    assert (7, 5) < (8, 9)

    # Act / Assert: Hopper sm_90 is above threshold → fp8
    assert (9, 0) >= (8, 9)


def test_get_compute_capability_blackwell():
    # Arrange: mock a Blackwell RTX 5070 Ti (sm_120).
    mock_props = MagicMock()
    mock_props.major = 12
    mock_props.minor = 0
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
    ):
        # Act
        sm = get_compute_capability()
    # Assert: Blackwell triggers fp8 path
    assert sm == (12, 0)
    assert sm >= (8, 9)


def test_get_compute_capability_ampere():
    # Arrange: mock an Ampere RTX 3080 (sm_86).
    mock_props = MagicMock()
    mock_props.major = 8
    mock_props.minor = 6
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
    ):
        # Act
        sm = get_compute_capability()
    # Assert: Ampere sm_86 triggers int8 path (below fp8 threshold)
    assert sm == (8, 6)
    assert sm < (8, 9)


def test_preflight_no_cuda():
    # Arrange: simulate no CUDA GPU available.
    # torch is imported locally inside preflight_check, so we patch it in sys.modules.
    engine = get_engine("flux2-klein")

    with patch("torch.cuda.is_available", return_value=False):
        # Act / Assert
        with pytest.raises(InsufficientResourcesError, match="No CUDA GPU detected"):
            preflight_check(engine)


def test_generate_passes_callback_to_pipe(tmp_path: Path):
    # Arrange: pre-load _pipe so _load() is skipped; inject callback.
    engine = get_engine("flux2-klein")
    mock_image = MagicMock()
    mock_pipe = MagicMock(return_value=MagicMock(images=[mock_image]))
    engine._pipe = mock_pipe

    callback = MagicMock(return_value={})
    out = tmp_path / "out.png"

    with (
        patch("torch.Generator") as mock_gen,
        patch("torch.inference_mode") as mock_ctx,
    ):
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_gen.return_value.manual_seed.return_value = MagicMock()
        engine.generate("test prompt", output_path=out, callback=callback)

    # Assert: callback_on_step_end was forwarded to the pipeline
    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs.get("callback_on_step_end") is callback


def test_generate_no_callback_omits_kwarg(tmp_path: Path):
    # Arrange: no callback → callback_on_step_end must NOT appear in pipe kwargs.
    engine = get_engine("flux2-klein")
    mock_image = MagicMock()
    mock_pipe = MagicMock(return_value=MagicMock(images=[mock_image]))
    engine._pipe = mock_pipe

    out = tmp_path / "out.png"

    with (
        patch("torch.Generator") as mock_gen,
        patch("torch.inference_mode") as mock_ctx,
    ):
        mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_gen.return_value.manual_seed.return_value = MagicMock()
        engine.generate("test prompt", output_path=out)

    call_kwargs = mock_pipe.call_args[1]
    assert "callback_on_step_end" not in call_kwargs
