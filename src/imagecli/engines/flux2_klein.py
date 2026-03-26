"""FLUX.2-klein-4B engine — default, bf16, fits in ~15GB VRAM."""

from __future__ import annotations

import logging

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)


class Flux2KleinEngine(ImageEngine):
    name = "flux2-klein"
    description = "FLUX.2-klein-4B — best quality for 16GB VRAM (Black Forest Labs, Nov 2025)"
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    vram_gb = 12.0
    capabilities = EngineCapabilities(negative_prompt=False)

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import Flux2KleinPipeline

        logger.info("Loading %s...", self.model_id)
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self._finalize_load(self._pipe)
        logger.info("Model ready.")
