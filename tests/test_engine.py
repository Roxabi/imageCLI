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
from imagecli.lora_spec import LoraSpec


# ── DummyEngine — minimal concrete subclass for __init__ shim tests ──────────


class DummyEngine(ImageEngine):
    name = "dummy"
    description = "Test stub"
    model_id = "test/model"
    vram_gb = 8.0

    def _load(self) -> None:  # no-op
        pass


# ── ImageEngine.__init__ — loras= plural form ────────────────────────────────


def test_engine_init_loras_plural_two():
    # Arrange / Act
    engine = DummyEngine(loras=[LoraSpec("/a"), LoraSpec("/b", scale=0.5)])

    # Assert: loras list has 2 elements
    assert len(engine.loras) == 2
    assert engine.loras[0] == LoraSpec("/a")
    assert engine.loras[1] == LoraSpec("/b", scale=0.5)

    # Singular attrs are None/1.0 when N != 1
    assert engine.lora_path is None
    assert engine.lora_scale == 1.0
    assert engine.trigger is None
    assert engine.embedding_path is None


def test_engine_init_loras_plural_empty():
    # Arrange / Act
    engine = DummyEngine(loras=[])

    # Assert: empty list is accepted; singular attrs are default
    assert engine.loras == []
    assert engine.lora_path is None
    assert engine.lora_scale == 1.0
    assert engine.trigger is None
    assert engine.embedding_path is None


# ── ImageEngine.__init__ — singular form ─────────────────────────────────────


def test_engine_init_singular_lora_path_only():
    # Arrange / Act
    engine = DummyEngine(lora_path="/a")

    # Assert: loras is a one-element list built from the singular kwargs
    assert engine.loras == [LoraSpec("/a", 1.0, None, None)]

    # Singular backward-compat attrs must be populated from that spec
    assert engine.lora_path == "/a"
    assert engine.lora_scale == 1.0
    assert engine.trigger is None
    assert engine.embedding_path is None


def test_engine_init_singular_all_fields():
    # Arrange / Act
    engine = DummyEngine(lora_path="/a", lora_scale=0.5, trigger="t", embedding_path="/e")

    # Assert: spec is built from all four singular kwargs
    assert engine.loras == [LoraSpec("/a", 0.5, "t", "/e")]
    assert engine.lora_path == "/a"
    assert engine.lora_scale == 0.5
    assert engine.trigger == "t"
    assert engine.embedding_path == "/e"


# ── ImageEngine.__init__ — no args ───────────────────────────────────────────


def test_engine_init_no_args():
    # Arrange / Act
    engine = DummyEngine()

    # Assert: empty loras, all singular attrs default
    assert engine.loras == []
    assert engine.lora_path is None
    assert engine.lora_scale == 1.0
    assert engine.trigger is None
    assert engine.embedding_path is None


# ── ImageEngine.__init__ — _UNSET sentinel semantics ─────────────────────────


def test_engine_init_lora_scale_unset_does_not_trigger_mixed_guard():
    # Arrange: loras= passed, lora_scale NOT passed → _UNSET → no mixed-form violation
    engine = DummyEngine(loras=[LoraSpec("/a")])

    # Assert: no ValueError raised; loras populated correctly
    assert engine.loras == [LoraSpec("/a")]


def test_engine_init_lora_scale_explicit_one_with_loras_raises():
    # Arrange: lora_scale=1.0 passed EXPLICITLY alongside loras= → mixed form
    # _UNSET sentinel exists to detect this case; lora_scale=1.0 is NOT the same
    # as lora_scale not passed.
    with pytest.raises(ValueError, match="loras="):
        DummyEngine(loras=[LoraSpec("/a")], lora_scale=1.0)  # type: ignore[call-overload]


# ── ImageEngine.__init__ — mixed-form raises ValueError ──────────────────────


def test_engine_init_mixed_form_loras_and_lora_path_raises():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="loras="):
        DummyEngine(loras=[LoraSpec("/a")], lora_path="/x")  # type: ignore[call-overload]


def test_engine_init_mixed_form_loras_and_trigger_raises():
    with pytest.raises(ValueError, match="loras="):
        DummyEngine(loras=[LoraSpec("/a")], trigger="t")  # type: ignore[call-overload]


def test_engine_init_mixed_form_loras_and_embedding_path_raises():
    with pytest.raises(ValueError, match="loras="):
        DummyEngine(loras=[LoraSpec("/a")], embedding_path="/e")  # type: ignore[call-overload]


def test_engine_init_mixed_form_error_message_mentions_both_forms():
    # Assert: error message mentions both loras= and the singular fields
    with pytest.raises(ValueError) as exc_info:
        DummyEngine(loras=[LoraSpec("/a")], lora_path="/x")  # type: ignore[call-overload]

    msg = str(exc_info.value)
    # Message should guide the user toward one form or the other
    assert "loras=" in msg or "lora_path" in msg


def test_registry_has_all_engines():
    # Act
    registry = _get_registry()

    # Assert
    assert "flux2-klein" in registry
    assert "flux1-dev" in registry
    assert "flux1-schnell" in registry
    assert "sd35" in registry
    assert "pulid-flux2-klein-fp4" in registry
    assert "pulid-flux2-klein" in registry
    assert "flux2-klein-fp4" in registry
    assert "flux2-klein-fp8" in registry
    assert "pulid-flux1-dev" in registry


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


def test_get_engine_rejects_loras_and_lora_path_combo():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="not both"):
        get_engine("flux2-klein", loras=[LoraSpec("/a")], lora_path="/x")


def test_get_engine_rejects_loras_and_trigger_combo():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="not both"):
        get_engine("flux2-klein", loras=[LoraSpec("/a")], trigger="t")


def test_get_engine_rejects_loras_and_embedding_path_combo():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="not both"):
        get_engine("flux2-klein", loras=[LoraSpec("/a")], embedding_path="/e")


def test_get_engine_accepts_loras_only():
    # Arrange
    specs = [LoraSpec("/a")]

    # Act
    engine = get_engine("flux2-klein", loras=specs)

    # Assert
    assert isinstance(engine, ImageEngine)
    assert engine.loras == specs


def test_get_engine_legacy_kwargs_emits_deprecation_warning():
    # Act / Assert: legacy singular kwargs fire DeprecationWarning
    with pytest.warns(DeprecationWarning, match="singular LoRA kwargs"):
        get_engine("flux2-klein", lora_path="/nonexistent.safetensors")


def test_get_engine_loras_does_not_warn():
    # Act: new plural form must not fire any warning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        get_engine("flux2-klein", loras=[LoraSpec("/a")])


def test_get_engine_no_lora_does_not_warn():
    # Act: plain call (no LoRA at all) must not fire the deprecation warning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        get_engine("flux2-klein")


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
    from typing import cast

    threshold = cast(tuple[int, int], (8, 9))

    # Act / Assert: Ada Lovelace boundary — exactly at threshold → fp8
    assert cast(tuple[int, int], (8, 9)) >= threshold

    # Act / Assert: Ampere sm_86 is below threshold → int8
    assert cast(tuple[int, int], (8, 6)) < threshold

    # Act / Assert: Blackwell sm_120 is above threshold → fp8
    assert cast(tuple[int, int], (12, 0)) >= threshold

    # Act / Assert: Turing sm_75 is below threshold → int8
    assert cast(tuple[int, int], (7, 5)) < threshold

    # Act / Assert: Hopper sm_90 is above threshold → fp8
    assert cast(tuple[int, int], (9, 0)) >= threshold


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
    assert set(call_kwargs.keys()) == {
        "prompt",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "generator",
    }


# ── T12: EngineCapabilities declarations ────────────────────────────────────


def test_engine_capabilities_dataclass():
    # Arrange / Act
    from imagecli.engine import EngineCapabilities

    caps = EngineCapabilities()

    # Assert: defaults match spec
    assert caps.negative_prompt is True
    assert caps.fixed_steps is None
    assert caps.fixed_guidance is None


def test_image_engine_default_capabilities():
    # Arrange / Act
    from imagecli.engine import EngineCapabilities, ImageEngine

    # Assert: base class has capabilities class attribute with correct defaults
    assert hasattr(ImageEngine, "capabilities")
    assert isinstance(ImageEngine.capabilities, EngineCapabilities)
    assert ImageEngine.capabilities.negative_prompt is True
    assert ImageEngine.capabilities.fixed_steps is None  # ADD THIS
    assert ImageEngine.capabilities.fixed_guidance is None  # ADD THIS


def test_flux2_klein_capabilities():
    # Arrange / Act
    from imagecli.engines.flux2_klein import Flux2KleinEngine

    # Assert: flux2-klein does NOT support negative_prompt; steps/guidance not fixed
    assert Flux2KleinEngine.capabilities.negative_prompt is False
    assert Flux2KleinEngine.capabilities.fixed_steps is None
    assert Flux2KleinEngine.capabilities.fixed_guidance is None


def test_flux1_dev_capabilities():
    # Arrange / Act
    from imagecli.engines.flux1_dev import Flux1DevEngine

    # Assert: flux1-dev does NOT support negative_prompt; steps/guidance not fixed
    assert Flux1DevEngine.capabilities.negative_prompt is False
    assert Flux1DevEngine.capabilities.fixed_steps is None
    assert Flux1DevEngine.capabilities.fixed_guidance is None


def test_flux1_schnell_capabilities():
    # Arrange / Act
    from imagecli.engines.flux1_schnell import Flux1SchnellEngine

    # Assert: flux1-schnell does NOT support negative_prompt; steps NOT fixed, guidance fixed at 0.0
    assert Flux1SchnellEngine.capabilities.negative_prompt is False
    assert Flux1SchnellEngine.capabilities.fixed_steps is None  # steps NOT fixed
    assert Flux1SchnellEngine.capabilities.fixed_guidance == 0.0


def test_sd35_capabilities():
    # Arrange / Act
    from imagecli.engines.sd35 import SD35Engine

    # Assert: sd35 DOES support negative_prompt; steps fixed at 20, guidance fixed at 1.0
    assert SD35Engine.capabilities.negative_prompt is True  # sd35 supports it
    assert SD35Engine.capabilities.fixed_steps == 20
    assert SD35Engine.capabilities.fixed_guidance == 1.0


# ── T13: warn_ignored_params ─────────────────────────────────────────────────


def test_warn_negative_prompt_flux2_klein(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("flux2-klein")

    # Act
    warn_ignored_params(
        engine,
        "ugly blurry",
        50,
        4.0,
        negative_explicit=True,
        steps_explicit=False,
        guidance_explicit=False,
    )
    captured = capsys.readouterr()

    # Assert: warning about negative_prompt printed to stderr
    assert "negative_prompt" in captured.err


def test_warn_negative_prompt_flux1_schnell(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("flux1-schnell")

    # Act
    warn_ignored_params(
        engine,
        "ugly",
        4,
        0.0,
        negative_explicit=True,
        steps_explicit=False,
        guidance_explicit=False,
    )
    captured = capsys.readouterr()

    # Assert: warning about negative_prompt printed to stderr
    assert "negative_prompt" in captured.err


def test_warn_steps_sd35_explicit(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("sd35")

    # Act: user explicitly set steps=30 but sd35 fixes them at 20
    warn_ignored_params(
        engine, "", 30, 4.0, negative_explicit=False, steps_explicit=True, guidance_explicit=False
    )
    captured = capsys.readouterr()

    # Assert: warning about steps printed to stderr
    assert "steps" in captured.err


def test_warn_guidance_sd35_explicit(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("sd35")

    # Act: user explicitly set guidance=7.5 but sd35 fixes it at 1.0
    warn_ignored_params(
        engine, "", 50, 7.5, negative_explicit=False, steps_explicit=False, guidance_explicit=True
    )
    captured = capsys.readouterr()

    # Assert: warning about guidance printed to stderr
    assert "guidance" in captured.err


def test_warn_guidance_flux1_schnell_explicit(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("flux1-schnell")

    # Act: user explicitly set guidance=7.5 but schnell fixes it at 0.0
    warn_ignored_params(
        engine, "", 4, 7.5, negative_explicit=False, steps_explicit=False, guidance_explicit=True
    )
    captured = capsys.readouterr()

    # Assert: warning about guidance printed to stderr
    assert "guidance" in captured.err


def test_no_warn_steps_flux2_klein_supported(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("flux2-klein")

    # Act: steps are supported on flux2-klein — no warning expected
    warn_ignored_params(
        engine, "", 30, 4.0, negative_explicit=False, steps_explicit=True, guidance_explicit=False
    )
    captured = capsys.readouterr()

    # Assert: no warning about steps
    assert "steps" not in captured.err


def test_no_warn_guidance_schnell_matches_fixed(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("flux1-schnell")

    # Act: user explicitly set guidance=0.0 which matches the fixed value — no warning
    warn_ignored_params(
        engine, "", 4, 0.0, negative_explicit=False, steps_explicit=False, guidance_explicit=True
    )
    captured = capsys.readouterr()

    # Assert: no warning about guidance when value matches fixed value
    assert "guidance" not in captured.err


def test_no_warn_steps_sd35_not_explicit(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("sd35")

    # Act: steps=50 would conflict with fixed_steps=20, but steps_explicit=False (came from config)
    warn_ignored_params(
        engine, "", 50, 4.0, negative_explicit=False, steps_explicit=False, guidance_explicit=False
    )
    captured = capsys.readouterr()

    # Assert: no warning when steps were not explicitly set by the user
    assert "steps" not in captured.err


def test_no_warn_empty_negative_flux2_klein(capsys):
    # Arrange
    from imagecli.engine import get_engine, warn_ignored_params

    engine = get_engine("flux2-klein")

    # Act: empty negative_prompt — nothing to warn about
    warn_ignored_params(
        engine, "", 50, 4.0, negative_explicit=False, steps_explicit=False, guidance_explicit=False
    )
    captured = capsys.readouterr()

    # Assert: no output at all on stderr
    assert captured.err == ""


# ── T14: list_engines capabilities key ──────────────────────────────────────


def test_list_engines_has_capabilities_key():
    # Act
    engines = list_engines()

    # Assert: every engine dict includes a 'capabilities' key with the expected shape
    for e in engines:
        assert "capabilities" in e, f"Engine {e['name']} missing 'capabilities' key"
        caps = e["capabilities"]
        assert "negative_prompt" in caps
        assert "fixed_steps" in caps
        assert "fixed_guidance" in caps


def test_list_engines_sd35_capabilities_values():
    # Act
    engines = {e["name"]: e for e in list_engines()}
    sd35 = engines["sd35"]["capabilities"]

    # Assert: sd35 capability values match spec
    assert sd35["fixed_steps"] == 20
    assert sd35["fixed_guidance"] == 1.0
    assert sd35["negative_prompt"] is True


def test_list_engines_flux2_klein_capabilities_values():
    # Act
    engines = {e["name"]: e for e in list_engines()}
    caps = engines["flux2-klein"]["capabilities"]

    # Assert: flux2-klein capability values match spec
    assert caps["negative_prompt"] is False
    assert caps["fixed_steps"] is None


# ── PuLIDFlux2KleinFP4Engine — _check_requirements() ───────────────────────


def test_pulid_fp4_check_requirements_no_cuda():
    """_check_requirements() raises RuntimeError when no CUDA GPU is present."""
    from imagecli.engines.pulid_flux2_klein_fp4 import PuLIDFlux2KleinFP4Engine

    engine = PuLIDFlux2KleinFP4Engine()

    with (
        patch.dict("sys.modules", {"comfy_kitchen": MagicMock()}),
        patch("torch.cuda.is_available", return_value=False),
    ):
        with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
            engine._check_requirements()


def test_pulid_fp4_check_requirements_wrong_sm():
    """_check_requirements() raises RuntimeError when GPU is below sm_120 (Blackwell)."""
    import torch as _torch
    from imagecli.engines.pulid_flux2_klein_fp4 import PuLIDFlux2KleinFP4Engine

    engine = PuLIDFlux2KleinFP4Engine()

    with (
        patch.dict("sys.modules", {"comfy_kitchen": MagicMock()}),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(8, 6)),
        patch.object(_torch.version, "cuda", "13.0"),
    ):
        with pytest.raises(RuntimeError, match="sm_120"):
            engine._check_requirements()


def test_pulid_fp4_check_requirements_old_cuda():
    """_check_requirements() raises RuntimeError when CUDA version is below 13."""
    import torch as _torch
    from imagecli.engines.pulid_flux2_klein_fp4 import PuLIDFlux2KleinFP4Engine

    engine = PuLIDFlux2KleinFP4Engine()

    with (
        patch.dict("sys.modules", {"comfy_kitchen": MagicMock()}),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        patch.object(_torch.version, "cuda", "12.4"),
    ):
        with pytest.raises(RuntimeError, match="CUDA 13.0"):
            engine._check_requirements()


def test_pulid_fp4_check_requirements_missing_comfy_kitchen():
    """_check_requirements() raises RuntimeError when comfy_kitchen is not importable."""
    import torch as _torch
    from imagecli.engines.pulid_flux2_klein_fp4 import PuLIDFlux2KleinFP4Engine

    engine = PuLIDFlux2KleinFP4Engine()

    with (
        patch.dict("sys.modules", {"comfy_kitchen": None}),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        patch.object(_torch.version, "cuda", "13.0"),
    ):
        with pytest.raises(RuntimeError, match="comfy-kitchen"):
            engine._check_requirements()


# ── PuLIDFlux2KleinFP4Engine — generate() face_image guard ──────────────────


def test_pulid_fp4_generate_requires_face_image():
    """generate() must raise ValueError when face_image is not provided."""
    from imagecli.engines.pulid_flux2_klein_fp4 import PuLIDFlux2KleinFP4Engine

    engine = PuLIDFlux2KleinFP4Engine()
    engine._pipe = MagicMock()  # skip _load()

    with pytest.raises(ValueError, match="face_image"):
        engine.generate("a prompt", output_path=Path("/tmp/out.png"))
