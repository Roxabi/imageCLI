"""Concrete base for 2-phase quant engines (flux2-klein family).

Finalizes the encode/generate plumbing that was previously duplicated word-for-word
across `flux2_klein.py`, `flux2_klein_fp8.py`, and `flux2_klein_fp4.py`. Subclasses
provide only `_load_pipeline()`; the rest is inherited.
"""

from __future__ import annotations

import gc
import logging
import random
from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from imagecli.engine import ImageEngine

logger = logging.getLogger(__name__)


class TwoPhaseBase(ImageEngine):
    """Concrete 2-phase batch base — overrides `TwoPhaseMixin` stubs with real impls.

    Subclasses must override `_load_pipeline()` to load + (optionally) quantize the
    underlying pipeline. The rest of the 2-phase + all-on-GPU machinery is inherited
    finalized.
    """

    supports_two_phase: ClassVar[bool] = True

    @abstractmethod
    def _load_pipeline(self) -> None:
        """Load the pipeline (from_pretrained + quantize). Must set ``self._pipe``."""

    def _teardown_encoder_phase(self) -> None:
        """Offload text encoder to CPU + free VRAM. Called between Phase 1 and Phase 2."""
        import torch

        assert self._pipe is not None
        self._pipe.text_encoder.to("cpu")  # type: ignore[attr-defined]
        torch.cuda.empty_cache()
        gc.collect()

    def encode_and_generate(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        guidance: float = 4.0,
        seed: int | None = None,
        output_path: Path,
        callback=None,
    ) -> Path:
        """Encode + generate in one shot (all-on-GPU mode)."""
        import torch

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pipe_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if callback is not None:
            pipe_kwargs["callback_on_step_end"] = callback

        assert self._pipe is not None
        with torch.inference_mode():
            result = self._pipe(**pipe_kwargs)  # type: ignore[operator]

        image = result.images[0]
        return self._save_image(
            image,
            output_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
        )

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
        callback=None,
    ) -> Path:
        """Generate image from pre-computed prompt embeddings (Phase 2)."""
        import torch

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        pipe_kwargs = {
            "prompt_embeds": embeddings["prompt_embeds"].to("cuda"),
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if callback is not None:
            pipe_kwargs["callback_on_step_end"] = callback

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        assert self._pipe is not None
        with torch.inference_mode():
            result = self._pipe(**pipe_kwargs)  # type: ignore[operator]

        image = result.images[0]
        return self._save_image(
            image,
            output_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
        )
