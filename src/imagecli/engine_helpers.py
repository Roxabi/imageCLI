"""Standalone helpers for :mod:`imagecli.engine_base`.

Keeps small, easily-testable pieces out of the ABC module so it stays
focused on ``ImageEngine`` + ``preflight_check``.

The module-level environment side-effects (RAM cap, TF32 flag) live here
so they execute exactly once regardless of which consumer imports first.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from imagecli.engine_base import ImageEngine

logger = logging.getLogger(__name__)

# Minimum free system RAM (in GB) required to start loading a model.
MIN_FREE_RAM_GB = float(os.environ.get("IMAGECLI_MIN_FREE_RAM_GB", "4.0"))

# Virtual address space cap — disabled by default.
# Models like FLUX.1 need ~24 GB virtual address space during quantization
# (bf16 weights in RAM before they're quantized and moved to GPU).
# Set IMAGECLI_MAX_RAM_GB to a positive value to re-enable.
_max_ram_gb = float(os.environ.get("IMAGECLI_MAX_RAM_GB", "0"))
if _max_ram_gb > 0:
    import resource as _resource

    _new_limit = int(_max_ram_gb * 1024**3)
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_AS)
    _cap = _new_limit if _hard == _resource.RLIM_INFINITY else min(_new_limit, _hard)
    _resource.setrlimit(_resource.RLIMIT_AS, (_cap, _hard))

# Process-global flag — set_float32_matmul_precision only needs to be called once.
_tf32_set = False


@dataclass
class EngineCapabilities:
    """Declares which generation parameters this engine actually uses."""

    negative_prompt: bool = True
    fixed_steps: int | None = None
    fixed_guidance: float | None = None


class InsufficientResourcesError(RuntimeError):
    """Raised when there isn't enough VRAM or system RAM to safely load an engine."""


def get_compute_capability() -> tuple[int, int]:
    """Return (major, minor) CUDA compute capability of GPU 0, or (0, 0) if unavailable."""
    import torch

    if not torch.cuda.is_available():
        return (0, 0)
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor)


def warn_ignored_params(
    engine: ImageEngine,
    negative_prompt: str,
    steps: int,
    guidance: float,
    *,
    negative_explicit: bool,
    steps_explicit: bool,
    guidance_explicit: bool,
) -> None:
    """Warn when the user provided a parameter the engine will ignore."""
    from rich.console import Console

    stderr = Console(stderr=True)
    caps = engine.capabilities

    if negative_explicit and not caps.negative_prompt:
        stderr.print(
            f"[yellow]Warning:[/yellow] {engine.name} ignores negative_prompt "
            f"(not supported by this engine)."
        )
    if steps_explicit and caps.fixed_steps is not None and steps != caps.fixed_steps:
        stderr.print(
            f"[yellow]Warning:[/yellow] {engine.name} ignores steps (fixed at {caps.fixed_steps})."
        )
    if guidance_explicit and caps.fixed_guidance is not None and guidance != caps.fixed_guidance:
        stderr.print(
            f"[yellow]Warning:[/yellow] {engine.name} ignores guidance "
            f"(fixed at {caps.fixed_guidance})."
        )


def quantize_transformer(pipe: object, sm: tuple[int, int]) -> str:
    """Quantize the transformer: fp8 on sm≥(8,9), int8 on Ampere. Returns qtype used."""
    try:
        from optimum.quanto import freeze, quantize  # type: ignore[import-untyped]

        if sm >= (8, 9):
            from optimum.quanto import qfloat8  # type: ignore[import-untyped]

            quantize(pipe.transformer, weights=qfloat8)
            qtype = "fp8"
        else:
            from optimum.quanto import qint8  # type: ignore[import-untyped]

            quantize(pipe.transformer, weights=qint8)
            qtype = "int8"
        freeze(pipe.transformer)
        return qtype
    except ImportError as e:
        raise RuntimeError(
            f"optimum-quanto is required for quantization: {e}\n"
            "Install it with: uv add optimum[quanto]"
        ) from e
    except (RuntimeError, ValueError) as e:
        logger.warning("Quantization failed (%s), proceeding with bf16 (~20GB VRAM required).", e)
        return "bf16"


class TwoPhaseMixin:
    """Optional 2-phase batch interface.

    Override in engines that benefit from splitting encoding and generation
    into separate GPU phases (e.g. flux2-klein). Default implementations
    raise ``NotImplementedError``.
    """

    name: str
    supports_two_phase: ClassVar[bool] = False

    def load_for_encode(self) -> None:
        raise NotImplementedError(f"{self.name} does not support 2-phase batch")

    def encode_prompt(self, prompt: str) -> dict:
        raise NotImplementedError(f"{self.name} does not support 2-phase batch")

    def start_generation_phase(self) -> None:
        raise NotImplementedError(f"{self.name} does not support 2-phase batch")

    def generate_from_embeddings(
        self,
        embeddings: dict,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        guidance: float = 4.0,
        seed: int | None = None,
        output_path: Path,
        callback: Callable[..., dict] | None = None,
    ) -> Path:
        raise NotImplementedError(f"{self.name} does not support 2-phase batch")
