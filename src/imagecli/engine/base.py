"""Abstract :class:`ImageEngine` base class."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from . import helpers as _h
from .helpers import (
    EngineCapabilities,
    InsufficientResourcesError,
    TwoPhaseMixin,
    logger,
    quantize_transformer,
)
from imagecli.lora_spec import LoraSpec


class ImageEngine(TwoPhaseMixin, ABC):
    name: str
    description: str
    model_id: str
    vram_gb: float  # approximate minimum VRAM
    capabilities: ClassVar[EngineCapabilities] = EngineCapabilities()

    def __init__(
        self,
        *,
        compile: bool = True,
        loras: list[LoraSpec] | None = None,
        lora_path: str | None = None,
        lora_scale: float | object = _h.UNSET,
        trigger: str | None = None,
        embedding_path: str | None = None,
    ) -> None:
        self._pipe: object | None = None
        self._compile = compile
        self._compiled = False
        self.loras: list[LoraSpec] = _h.resolve_loras(
            loras, lora_path, lora_scale, trigger, embedding_path
        )

        # Backward-compat singular attrs. The one-element case preserves legacy
        # behavior for subclasses that still read them. Multi-LoRA leaves them
        # None/1.0 — no single value is meaningful, and callers must migrate.
        if len(self.loras) == 1:
            spec = self.loras[0]
            self.lora_path = spec.path
            self.lora_scale = spec.scale
            self.trigger = spec.trigger
            self.embedding_path = spec.embedding_path
        else:
            self.lora_path = None
            self.lora_scale = 1.0
            self.trigger = None
            self.embedding_path = None

    @abstractmethod
    def _load(self) -> None:
        """Load the model pipeline into self._pipe. Called lazily on first generate()."""

    def effective_steps(self, requested: int) -> int:
        """Return the step count that will actually run. Override for engines with fixed steps."""
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
        import random

        import torch

        self._load()

        # Auto-generate seed when none provided — ensures reproducibility
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cuda").manual_seed(seed)

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
            assert callable(self._pipe), "Pipeline must be loaded before generate()"
            result = self._pipe(**pipe_kwargs)

        image = result.images[0]  # type: ignore[union-attr]
        return self._save_image(
            image,
            output_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
        )

    def _save_image(
        self,
        image,
        output_path: Path,
        *,
        seed: int,
        steps: int,
        guidance: float,
        width: int,
        height: int,
    ) -> Path:
        return _h.save_image(
            image, output_path, engine_name=self.name,
            seed=seed, steps=steps, guidance=guidance, width=width, height=height,
        )

    def _finalize_load(self, pipe: object) -> None:
        """Move pipeline to GPU and apply optimizations. Raises if VRAM is insufficient."""
        import torch

        if torch.cuda.is_available():
            free_vram_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
            if free_vram_gb < self.vram_gb:
                raise InsufficientResourcesError(
                    f"Engine {self.name!r} needs ~{self.vram_gb:.1f} GB VRAM, "
                    f"but only {free_vram_gb:.1f} GB is free. "
                    f"Close other GPU processes and retry."
                )
            pipe.to("cuda")  # type: ignore[union-attr]
        self._optimize_pipe(pipe)

    def _optimize_pipe(self, pipe: object, *, compile: bool | None = None) -> None:
        """Apply performance optimizations. compile=False overrides self._compile.

        Use compile=False when CPU offload hooks are active, since torch.compile
        is incompatible with enable_model_cpu_offload.
        """
        import torch

        # TF32 matmul — tensor cores on Blackwell (~10-15% speedup, once per process)
        if not _h._tf32_set:
            torch.set_float32_matmul_precision("high")
            _h._tf32_set = True

        # VAE optimizations — reduce peak VRAM for large images and batches
        if hasattr(pipe, "vae"):
            pipe.vae.enable_tiling()  # type: ignore[union-attr]
            pipe.vae.enable_slicing()  # type: ignore[union-attr]

        # torch.compile — 20-40% speedup after first-run warmup
        should_compile = self._compile if compile is None else compile
        if should_compile and not self._compiled:
            compilable = None
            if hasattr(pipe, "transformer"):
                compilable = pipe.transformer  # type: ignore[union-attr]
            elif hasattr(pipe, "unet"):
                compilable = pipe.unet  # type: ignore[union-attr]
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
        """Quantize the transformer: fp8 on sm≥(8,9), int8 on Ampere."""
        return quantize_transformer(pipe, sm)

    def clear_cache(self) -> None:
        """Clear CUDA cache between generations without unloading the model."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def cleanup(self) -> None:
        """Fully unload model and free all VRAM / RAM."""
        import torch

        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._compiled = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _apply_pivotal_embeddings(self) -> None:
        """Load and apply pivotal tuning embeddings for each LoRA in ``self.loras``.

        No-op when ``self.loras`` is empty. Must be called AFTER all LoRAs have
        been fused/unloaded and BEFORE the transformer is quantized or moved
        to GPU. Tokenizer mutations from all pivotals are applied atomically:
        either every trigger is added (and every vector written) or none are,
        so a collision on trigger #N does not leave #1..#N-1 dangling.
        """
        if not self.loras:
            return
        from imagecli.pivotal import (
            _patch_encode_prompt,
            apply_pivotals_to_pipe,
            load_pivotal_embedding,
        )

        assert self._pipe is not None, "Pipeline must be loaded before _apply_pivotal_embeddings()"
        te_hidden_size = self._pipe.text_encoder.config.hidden_size  # type: ignore[union-attr]
        pivotals = []
        for spec in self.loras:
            p = load_pivotal_embedding(
                spec.path,
                spec.trigger,
                embedding_path=spec.embedding_path,
                te_hidden_size=te_hidden_size,
            )
            if p is not None:
                pivotals.append(p)
        if not pivotals:
            return
        apply_pivotals_to_pipe(self._pipe, pivotals)
        _patch_encode_prompt(self._pipe)


