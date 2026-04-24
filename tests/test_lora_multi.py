"""Tests for V3 multi-LoRA fuse loop — Klein (quanto), Klein FP8, Klein FP4 guard.

T9  [RED]  — N=2 fuse loop for flux2-klein and flux2-klein-fp8
T13 [RED]  — N=1 byte-identical, FP4 guard, FP4 empty-loras passthrough

Strategy: patch the heavy I/O via sys.modules so _load_pipeline runs
end-to-end with a MagicMock pipe, then assert on the mock's call history.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

from imagecli.lora_spec import LoraSpec


# ── mock pipe factory ─────────────────────────────────────────────────────────


def _make_mock_pipe() -> MagicMock:
    """Minimal mock pipe that passes _apply_pivotal_embeddings guard checks."""
    pipe = MagicMock()
    pipe.text_encoder.config.hidden_size = 2560
    return pipe


# ── heavy-import stubs for Klein (quanto) ────────────────────────────────────


def _klein_sys_modules(pipe: MagicMock) -> dict:
    """Return sys.modules overrides that stub out all heavy imports for Klein."""
    mock_diffusers = MagicMock()
    mock_diffusers.Flux2KleinPipeline.from_pretrained.return_value = pipe

    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"

    mock_quanto = MagicMock()
    mock_quanto.nn = MagicMock()
    mock_quanto.nn.QLinear = MagicMock()
    mock_quanto.nn.QLinear.forward = MagicMock()

    return {
        "torch": mock_torch,
        "diffusers": mock_diffusers,
        "optimum": MagicMock(),
        "optimum.quanto": mock_quanto,
        "optimum.quanto.nn": mock_quanto.nn,
    }


def _fp8_sys_modules(pipe: MagicMock) -> dict:
    """Return sys.modules overrides for FP8 engine."""
    mock_diffusers = MagicMock()
    mock_diffusers.Flux2KleinPipeline.from_pretrained.return_value = pipe

    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"

    mock_torchao_quant = MagicMock()
    mock_torchao_quant.Float8WeightOnlyConfig = MagicMock()
    mock_torchao_quant.quantize_ = MagicMock()

    return {
        "torch": mock_torch,
        "diffusers": mock_diffusers,
        "torchao": MagicMock(),
        "torchao.quantization": mock_torchao_quant,
    }


def _run_klein_load_pipeline(engine, pipe: MagicMock) -> None:
    """Run Flux2KleinEngine._load_pipeline with all heavy ops stubbed."""
    with (
        patch.dict(sys.modules, _klein_sys_modules(pipe)),
        patch.object(engine, "_apply_pivotal_embeddings"),
    ):
        engine._load_pipeline()


def _run_fp8_load_pipeline(engine, pipe: MagicMock) -> None:
    """Run Flux2KleinFP8Engine._load_pipeline with all heavy ops stubbed."""
    with (
        patch.dict(sys.modules, _fp8_sys_modules(pipe)),
        patch.object(engine, "_apply_pivotal_embeddings"),
    ):
        engine._load_pipeline()


# ── flux2_klein — N=2 ────────────────────────────────────────────────────────


def _klein_n2_run():
    specs = [
        LoraSpec(path="/lora/a.safetensors", scale=0.8),
        LoraSpec(path="/lora/b.safetensors", scale=1.2),
    ]
    from imagecli.engines.flux2_klein import Flux2KleinEngine

    engine = Flux2KleinEngine(loras=specs)
    pipe = _make_mock_pipe()
    _run_klein_load_pipeline(engine, pipe)
    return pipe


def test_klein_n2_load_lora_called_twice():
    """Both load_lora_weights calls happen with distinct adapter_name values."""
    pipe = _klein_n2_run()
    calls = pipe.load_lora_weights.call_args_list
    assert len(calls) == 2
    assert calls[0] == call("/lora/a.safetensors", adapter_name="lora_0")
    assert calls[1] == call("/lora/b.safetensors", adapter_name="lora_1")


def test_klein_n2_set_adapters_both_names_and_weights():
    """set_adapters called once with 2-element name list and matching scales."""
    pipe = _klein_n2_run()
    pipe.set_adapters.assert_called_once_with(["lora_0", "lora_1"], adapter_weights=[0.8, 1.2])


def test_klein_n2_fuse_lora_once_with_scale_1():
    """fuse_lora called once with lora_scale=1.0 when N>1."""
    pipe = _klein_n2_run()
    pipe.fuse_lora.assert_called_once_with(lora_scale=1.0)


def test_klein_n2_unload_lora_weights_once():
    """unload_lora_weights called exactly once after fuse."""
    pipe = _klein_n2_run()
    pipe.unload_lora_weights.assert_called_once()


# ── flux2_klein — N=1 byte-identical ─────────────────────────────────────────


def _klein_n1_run(scale: float = 0.75):
    spec = LoraSpec(path="/lora/single.safetensors", scale=scale)
    from imagecli.engines.flux2_klein import Flux2KleinEngine

    engine = Flux2KleinEngine(loras=[spec])
    pipe = _make_mock_pipe()
    _run_klein_load_pipeline(engine, pipe)
    return pipe


def test_klein_n1_no_set_adapters_call():
    """N=1 must NOT call set_adapters (byte-identical to pre-#34 behavior)."""
    pipe = _klein_n1_run()
    pipe.set_adapters.assert_not_called()


def test_klein_n1_fuse_lora_uses_spec_scale():
    """N=1: fuse_lora(lora_scale=spec.scale) matches the pre-#34 scalar form."""
    pipe = _klein_n1_run(scale=0.75)
    pipe.fuse_lora.assert_called_once_with(lora_scale=0.75)


def test_klein_n1_load_lora_weights_called_once():
    """N=1: load_lora_weights called once with adapter_name='lora_0'."""
    pipe = _klein_n1_run()
    pipe.load_lora_weights.assert_called_once_with(
        "/lora/single.safetensors", adapter_name="lora_0"
    )


def test_klein_n1_unload_lora_weights_called():
    pipe = _klein_n1_run()
    pipe.unload_lora_weights.assert_called_once()


# ── flux2_klein — no loras ────────────────────────────────────────────────────


def test_klein_no_loras_skips_fuse_block():
    """Empty loras → no LoRA calls at all."""
    from imagecli.engines.flux2_klein import Flux2KleinEngine

    engine = Flux2KleinEngine(loras=[])
    pipe = _make_mock_pipe()
    _run_klein_load_pipeline(engine, pipe)

    pipe.load_lora_weights.assert_not_called()
    pipe.set_adapters.assert_not_called()
    pipe.fuse_lora.assert_not_called()
    pipe.unload_lora_weights.assert_not_called()


# ── flux2_klein_fp8 — N=2 mirrors Klein ──────────────────────────────────────


def _fp8_n2_run():
    specs = [
        LoraSpec(path="/lora/a.safetensors", scale=0.9),
        LoraSpec(path="/lora/b.safetensors", scale=1.1),
    ]
    from imagecli.engines.flux2_klein_fp8 import Flux2KleinFP8Engine

    engine = Flux2KleinFP8Engine(loras=specs)
    pipe = _make_mock_pipe()
    _run_fp8_load_pipeline(engine, pipe)
    return pipe


def test_fp8_n2_load_lora_called_twice():
    pipe = _fp8_n2_run()
    calls = pipe.load_lora_weights.call_args_list
    assert len(calls) == 2
    assert calls[0] == call("/lora/a.safetensors", adapter_name="lora_0")
    assert calls[1] == call("/lora/b.safetensors", adapter_name="lora_1")


def test_fp8_n2_set_adapters_names_and_weights():
    pipe = _fp8_n2_run()
    pipe.set_adapters.assert_called_once_with(["lora_0", "lora_1"], adapter_weights=[0.9, 1.1])


def test_fp8_n2_fuse_lora_scale_1():
    pipe = _fp8_n2_run()
    pipe.fuse_lora.assert_called_once_with(lora_scale=1.0)


def test_fp8_n2_unload_lora_weights_once():
    pipe = _fp8_n2_run()
    pipe.unload_lora_weights.assert_called_once()


# ── flux2_klein_fp8 — N=1 ────────────────────────────────────────────────────


def test_fp8_n1_no_set_adapters():
    """N=1 FP8: no set_adapters; fuse_lora uses spec scale."""
    spec = LoraSpec(path="/lora/a.safetensors", scale=0.5)
    from imagecli.engines.flux2_klein_fp8 import Flux2KleinFP8Engine

    engine = Flux2KleinFP8Engine(loras=[spec])
    pipe = _make_mock_pipe()
    _run_fp8_load_pipeline(engine, pipe)

    pipe.set_adapters.assert_not_called()
    pipe.fuse_lora.assert_called_once_with(lora_scale=0.5)


def test_fp8_no_loras_skips_fuse_block():
    from imagecli.engines.flux2_klein_fp8 import Flux2KleinFP8Engine

    engine = Flux2KleinFP8Engine(loras=[])
    pipe = _make_mock_pipe()
    _run_fp8_load_pipeline(engine, pipe)

    pipe.load_lora_weights.assert_not_called()
    pipe.set_adapters.assert_not_called()
    pipe.fuse_lora.assert_not_called()
    pipe.unload_lora_weights.assert_not_called()


# ── flux2_klein_fp4 — LoRA guard ─────────────────────────────────────────────


def test_fp4_raises_value_error_single_lora():
    """N=1: _load_pipeline raises ValueError immediately (pre-quantized guard)."""
    from imagecli.engines.flux2_klein_fp4 import Flux2KleinFP4Engine

    engine = Flux2KleinFP4Engine(loras=[LoraSpec(path="/lora/a.safetensors", scale=1.0)])

    with pytest.raises(ValueError, match="flux2-klein-fp4 does not support LoRA"):
        engine._load_pipeline()


def test_fp4_raises_value_error_multiple_loras():
    """N=2: guard fires for multiple LoRAs too."""
    from imagecli.engines.flux2_klein_fp4 import Flux2KleinFP4Engine

    engine = Flux2KleinFP4Engine(
        loras=[
            LoraSpec(path="/lora/a.safetensors", scale=1.0),
            LoraSpec(path="/lora/b.safetensors", scale=1.0),
        ]
    )

    with pytest.raises(ValueError, match="flux2-klein-fp4 does not support LoRA"):
        engine._load_pipeline()


def test_fp4_error_message_suggests_alternatives():
    """Error message must name valid alternative engines."""
    from imagecli.engines.flux2_klein_fp4 import Flux2KleinFP4Engine

    engine = Flux2KleinFP4Engine(loras=[LoraSpec(path="/lora/a.safetensors")])

    with pytest.raises(ValueError, match="flux2-klein"):
        engine._load_pipeline()


def test_fp4_guard_does_not_fire_when_no_loras():
    """_load_pipeline with no loras must not raise ValueError.

    The guard fires only when self.loras is non-empty.  With empty loras the
    method proceeds into the real load path, which we stub out entirely.
    """
    import imagecli.engines.flux2_klein_fp4 as fp4_mod
    from imagecli.engines.flux2_klein_fp4 import Flux2KleinFP4Engine

    engine = Flux2KleinFP4Engine(loras=[])
    mock_pipe = _make_mock_pipe()

    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"
    mock_diffusers = MagicMock()
    mock_diffusers.Flux2KleinPipeline.from_pretrained.return_value = mock_pipe
    mock_hhub = MagicMock()
    mock_hhub.hf_hub_download.return_value = "/fake/nvfp4.safetensors"

    stub_modules = {
        "torch": mock_torch,
        "diffusers": mock_diffusers,
        "huggingface_hub": mock_hhub,
    }

    with (
        patch.dict(sys.modules, stub_modules),
        patch.object(engine, "_check_requirements"),
        patch.object(fp4_mod, "_patch_transformer_nvfp4", return_value=10),
        patch.object(engine, "_apply_pivotal_embeddings"),
    ):
        # No ValueError should be raised
        engine._load_pipeline()

    # Guard did not block; pipe was assigned
    assert engine._pipe is not None
