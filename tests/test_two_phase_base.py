"""Unit tests for engines._two_phase_base — finalized 2-phase plumbing.

Verifies:
- TwoPhaseBase inherits ImageEngine and overrides supports_two_phase to True
- All 3 quant engines inherit TwoPhaseBase
- encode_and_generate / generate_from_embeddings call self._pipe and self._save_image
- _teardown_encoder_phase offloads text_encoder + clears CUDA cache + gc.collect

No GPU / model download required — _pipe is a MagicMock.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from imagecli.engine import ImageEngine
from imagecli.engines._two_phase_base import TwoPhaseBase


class _FakeEngine(TwoPhaseBase):
    """Concrete TwoPhaseBase subclass for testing — stubs `_load` + `_load_pipeline`."""

    name = "fake-two-phase"
    description = "test stub"
    model_id = "fake/model"
    vram_gb: ClassVar[float] = 0.0  # type: ignore[misc]

    def _load(self) -> None:
        pass

    def _load_pipeline(self) -> None:
        pass  # tests inject _pipe directly


def _make_engine_with_pipe() -> tuple[_FakeEngine, MagicMock, MagicMock]:
    """Return (engine, pipe_mock, save_image_mock) with both wired."""
    eng = _FakeEngine()
    pipe = MagicMock()
    eng._pipe = pipe
    save_image = MagicMock(return_value=Path("/tmp/fake.png"))
    eng._save_image = save_image  # type: ignore[method-assign]
    return eng, pipe, save_image


# ── Inheritance chain ────────────────────────────────────────────────────────


def test_two_phase_base_inherits_image_engine():
    assert issubclass(TwoPhaseBase, ImageEngine)


def test_two_phase_base_supports_two_phase_true():
    assert TwoPhaseBase.supports_two_phase is True


@pytest.mark.parametrize(
    ("engine_module", "engine_class_name"),
    [
        ("imagecli.engines.flux2_klein", "Flux2KleinEngine"),
        ("imagecli.engines.flux2_klein_fp8", "Flux2KleinFP8Engine"),
        ("imagecli.engines.flux2_klein_fp4", "Flux2KleinFP4Engine"),
    ],
)
def test_quant_engines_inherit_two_phase_base(engine_module: str, engine_class_name: str):
    """Regression guard: each flux2-klein quant engine must keep inheriting TwoPhaseBase."""
    mod = importlib.import_module(engine_module)
    cls = getattr(mod, engine_class_name)
    assert issubclass(cls, TwoPhaseBase), f"{engine_class_name} must inherit TwoPhaseBase"
    assert cls.supports_two_phase is True


# ── encode_and_generate ──────────────────────────────────────────────────────


def test_encode_and_generate_calls_pipe_and_saves(tmp_path: Path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    eng, pipe, save_image = _make_engine_with_pipe()
    fake_result = MagicMock()
    fake_result.images = [MagicMock()]
    pipe.return_value = fake_result

    returned = eng.encode_and_generate(
        prompt="hello",
        width=512,
        height=512,
        steps=10,
        guidance=4.0,
        seed=42,
        output_path=tmp_path / "out.png",
    )

    assert pipe.called
    save_image.assert_called_once()
    assert returned == Path("/tmp/fake.png")
    kwargs = pipe.call_args.kwargs
    assert kwargs["prompt"] == "hello"
    assert kwargs["width"] == 512
    assert kwargs["num_inference_steps"] == 10
    assert kwargs["guidance_scale"] == 4.0


def test_encode_and_generate_passes_callback(tmp_path: Path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    eng, pipe, _ = _make_engine_with_pipe()
    fake_result = MagicMock()
    fake_result.images = [MagicMock()]
    pipe.return_value = fake_result

    cb = MagicMock()
    eng.encode_and_generate(
        prompt="hello",
        seed=1,
        output_path=tmp_path / "out.png",
        callback=cb,
    )
    assert pipe.call_args.kwargs["callback_on_step_end"] is cb


# ── generate_from_embeddings ─────────────────────────────────────────────────


def test_generate_from_embeddings_moves_embeds_to_cuda_and_saves(tmp_path: Path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    eng, pipe, save_image = _make_engine_with_pipe()
    fake_result = MagicMock()
    fake_result.images = [MagicMock()]
    pipe.return_value = fake_result

    embed = MagicMock()
    embed.to.return_value = embed

    eng.generate_from_embeddings(
        embeddings={"prompt_embeds": embed},
        width=512,
        height=512,
        steps=10,
        guidance=4.0,
        seed=42,
        output_path=tmp_path / "out.png",
    )

    embed.to.assert_called_with("cuda")
    assert pipe.called
    save_image.assert_called_once()


# ── _teardown_encoder_phase ──────────────────────────────────────────────────


def test_teardown_encoder_phase_offloads_and_clears(monkeypatch):
    import torch

    eng, pipe, _ = _make_engine_with_pipe()

    empty_cache_calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    gc_calls: list[bool] = []
    monkeypatch.setattr(
        "imagecli.engines._two_phase_base.gc.collect", lambda: gc_calls.append(True)
    )

    eng._teardown_encoder_phase()

    pipe.text_encoder.to.assert_called_with("cpu")
    assert empty_cache_calls == [True]
    assert gc_calls == [True]


def test_teardown_encoder_phase_requires_loaded_pipe():
    eng = _FakeEngine()
    # _pipe stays None — assert should fail loudly
    with pytest.raises(AssertionError):
        eng._teardown_encoder_phase()
