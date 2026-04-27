"""Standalone helpers for :mod:`imagecli.engine_base`.

Keeps small, easily-testable pieces out of the ABC module so it stays
focused on the ``ImageEngine`` ABC.

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
    from imagecli.lora_spec import LoraSpec  # noqa: F401 — used in _resolve_loras signature

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

            quantize(pipe.transformer, weights=qfloat8)  # type: ignore[union-attr]
            qtype = "fp8"
        else:
            from optimum.quanto import qint8  # type: ignore[import-untyped]

            quantize(pipe.transformer, weights=qint8)  # type: ignore[union-attr]
            qtype = "int8"
        freeze(pipe.transformer)  # type: ignore[union-attr]
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


# ── LoRA init helper ─────────────────────────────────────────────────────────

# Sentinel — distinguishes "lora_scale not passed" from "lora_scale=1.0 passed
# explicitly". Needed to detect mixed-form use alongside ``loras=``.
_UNSET: object = object()


def _resolve_loras(
    loras: list[LoraSpec] | None,
    lora_path: str | None,
    lora_scale: float | object,
    trigger: str | None,
    embedding_path: str | None,
) -> tuple[list[LoraSpec], tuple[str | None, float, str | None, str | None]]:
    """Resolve LoRA constructor args into (loras_list, singular_compat_tuple)."""
    from imagecli.lora_spec import LoraSpec as _LoraSpec

    singular_set = (
        lora_path is not None
        or trigger is not None
        or embedding_path is not None
        or lora_scale is not _UNSET
    )
    if loras is not None and singular_set:
        raise ValueError(
            "Pass either loras= or the singular fields "
            "(lora_path / lora_scale / trigger / embedding_path), not both."
        )

    if loras is not None:
        resolved: list[LoraSpec] = list(loras)
    elif lora_path is not None:
        resolved = [
            _LoraSpec(
                path=lora_path,
                scale=1.0 if lora_scale is _UNSET else float(lora_scale),  # type: ignore[arg-type]
                trigger=trigger,
                embedding_path=embedding_path,
            )
        ]
    elif embedding_path is not None or trigger is not None:
        raise ValueError(
            "embedding_path / trigger require lora_path (pivotal embeddings "
            "are scoped to a LoRA). Pass lora_path=... or use loras=[LoraSpec(...)]."
        )
    else:
        resolved = []

    if len(resolved) == 1:
        spec = resolved[0]
        singular: tuple[str | None, float, str | None, str | None] = (
            spec.path,
            spec.scale,
            spec.trigger,
            spec.embedding_path,
        )
    else:
        singular = (None, 1.0, None, None)

    return resolved, singular


# ── System resource check ────────────────────────────────────────────────────


def preflight_check(engine: ImageEngine) -> None:
    """Abort early if the system can't safely run this engine. Skipped if already loaded."""
    if engine._pipe is not None:
        return

    import torch

    if torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        free_vram_gb = free_vram / 1024**3
        if free_vram_gb < engine.vram_gb:
            raise InsufficientResourcesError(
                f"Engine {engine.name!r} needs ~{engine.vram_gb:.1f} GB VRAM, "
                f"but only {free_vram_gb:.1f} GB / {total_vram / 1024**3:.1f} GB is free. "
                f"Close other GPU processes (e.g. ollama) and retry."
            )
    else:
        raise InsufficientResourcesError(
            "No CUDA GPU detected. imagecli requires a CUDA-capable GPU."
        )

    ram_needed_gb = max(MIN_FREE_RAM_GB, engine.vram_gb)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    available_gb = available_kb / 1024 / 1024
                    if available_gb < ram_needed_gb:
                        raise InsufficientResourcesError(
                            f"Only {available_gb:.1f} GB system RAM available, "
                            f"need at least {ram_needed_gb:.1f} GB for engine "
                            f"{engine.name!r}. Close other applications and retry."
                        )
                    break
    except FileNotFoundError:
        pass  # non-Linux, skip RAM check
