"""Abstract ImageEngine base class + engine registry."""

from __future__ import annotations

import gc
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

# Minimum free system RAM (in GB) required to start loading a model.
# model_cpu_offload shuffles layers through CPU, so we need headroom.
logger = logging.getLogger(__name__)

MIN_FREE_RAM_GB = float(os.environ.get("IMAGECLI_MIN_FREE_RAM_GB", "4.0"))

# Process-global flag — set_float32_matmul_precision only needs to be called once.
_tf32_set = False


class InsufficientResourcesError(RuntimeError):
    """Raised when there isn't enough VRAM or system RAM to safely load an engine."""


class ImageEngine(ABC):
    name: str
    description: str
    model_id: str
    vram_gb: float  # approximate minimum VRAM

    def __init__(self, *, compile: bool = True) -> None:
        self._pipe: object | None = None
        self._compile = compile
        self._compiled = False

    @abstractmethod
    def _load(self) -> None:
        """Load the model pipeline into self._pipe. Called lazily on first generate()."""

    def effective_steps(self, requested: int) -> int:
        """Return the number of inference steps that will actually run.

        The default honours the caller's requested value. Engines that hardcode
        a fixed step count (e.g. SD3.5 Turbo) override this so that progress
        bars and callers can display the correct total.
        """
        return requested

    def _build_pipe_kwargs(
        self,
        prompt: str,
        *,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        generator: object | None,
    ) -> dict:
        """Return kwargs for the pipeline call. Override for engine-specific args."""
        return {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        guidance: float = 4.0,
        seed: int | None = None,
        output_path: Path,
        callback: Callable[..., dict] | None = None,
        **kwargs,
    ) -> Path:
        """Generate an image and save it to output_path. Returns the saved path."""
        import torch

        self._load()

        generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None

        pipe_kwargs = self._build_pipe_kwargs(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            generator=generator,
        )
        if callback is not None:
            pipe_kwargs["callback_on_step_end"] = callback

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            result = self._pipe(**pipe_kwargs)

        image = result.images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path))
        return output_path

    def _finalize_load(self, pipe: object) -> None:
        """Common post-load setup: CPU offload, VRAM cap, optimizations."""
        import torch

        pipe.enable_model_cpu_offload(gpu_id=0)
        torch.cuda.set_per_process_memory_fraction(0.85)
        self._optimize_pipe(pipe)

    def _optimize_pipe(self, pipe: object) -> None:
        """Apply performance optimizations to a loaded pipeline."""
        global _tf32_set  # noqa: PLW0603
        import torch

        # TF32 matmul — uses tensor cores on Blackwell for ~10-15% speedup (once)
        if not _tf32_set:
            torch.set_float32_matmul_precision("high")
            _tf32_set = True

        # VAE optimizations — reduces peak VRAM for large images and batches
        if hasattr(pipe, "vae"):
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

        # torch.compile — fuses kernels for 20-40% speedup after first-run warmup
        if self._compile and not self._compiled:
            compilable = None
            if hasattr(pipe, "transformer"):
                compilable = pipe.transformer
            elif hasattr(pipe, "unet"):
                compilable = pipe.unet
            if compilable is not None:
                pipe_attr = "transformer" if hasattr(pipe, "transformer") else "unet"
                logger.info(
                    "Compiling %s with torch.compile (first run will be slower)...", pipe_attr
                )
                setattr(
                    pipe,
                    pipe_attr,
                    torch.compile(compilable, mode="max-autotune-no-cudagraphs"),
                )
                self._compiled = True

    def _quantize_transformer(self, pipe: object, sm: tuple[int, int]) -> str:
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
            logger.warning(
                "Quantization failed (%s), proceeding with bf16 (~20GB VRAM required).", e
            )
            return "bf16"

    def cleanup(self) -> None:
        """Free VRAM / RAM after generation."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def preflight_check(engine: ImageEngine) -> None:
    """Abort early if the system can't safely run this engine. Skipped if already loaded."""
    if engine._pipe is not None:
        return

    import torch

    # --- VRAM check ---
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

    # --- System RAM check ---
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
                    available_gb = available_kb / 1024 / 1024
                    if available_gb < MIN_FREE_RAM_GB:
                        raise InsufficientResourcesError(
                            f"Only {available_gb:.1f} GB system RAM available, "
                            f"need at least {MIN_FREE_RAM_GB:.1f} GB. "
                            f"Close other applications and retry."
                        )
                    break
    except FileNotFoundError:
        pass  # non-Linux, skip RAM check


def get_compute_capability() -> tuple[int, int]:
    """Return (major, minor) CUDA compute capability of GPU 0, or (0, 0) if unavailable."""
    import torch

    if not torch.cuda.is_available():
        return (0, 0)
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor)


def _get_registry() -> dict[str, type[ImageEngine]]:
    from imagecli.engines.flux1_dev import Flux1DevEngine
    from imagecli.engines.flux1_schnell import Flux1SchnellEngine
    from imagecli.engines.flux2_klein import Flux2KleinEngine
    from imagecli.engines.sd35 import SD35Engine

    return {
        "flux2-klein": Flux2KleinEngine,
        "flux1-dev": Flux1DevEngine,
        "flux1-schnell": Flux1SchnellEngine,
        "sd35": SD35Engine,
    }


def get_engine(name: str, *, compile: bool = True) -> ImageEngine:
    registry = _get_registry()
    if name not in registry:
        known = ", ".join(registry)
        raise ValueError(f"Unknown engine {name!r}. Available: {known}")
    return registry[name](compile=compile)


def list_engines() -> list[dict]:
    registry = _get_registry()
    return [
        {
            "name": name,
            "description": cls.description,
            "model_id": cls.model_id,
            "vram_gb": cls.vram_gb,
        }
        for name, cls in registry.items()
    ]
