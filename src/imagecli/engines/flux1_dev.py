"""FLUX.1-dev engine — quantized (fp8 on sm≥89, int8 on Ampere), ~10GB VRAM."""

from __future__ import annotations

import logging

from imagecli.engine import EngineCapabilities, ImageEngine, get_compute_capability

logger = logging.getLogger(__name__)


class Flux1DevEngine(ImageEngine):
    name = "flux1-dev"
    description = "FLUX.1-dev quantized — top quality, ~10GB VRAM (Black Forest Labs)"
    model_id = "black-forest-labs/FLUX.1-dev"
    vram_gb = 10.0
    capabilities = EngineCapabilities(negative_prompt=False)

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline  # type: ignore[import-untyped]

        sm = get_compute_capability()
        qtype_label = "fp8" if sm >= (8, 9) else "int8"
        logger.info("Loading %s (%s)...", self.model_id, qtype_label)
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        actual_qtype = self._quantize_transformer(self._pipe, sm)
        logger.info("Transformer quantized to %s.", actual_qtype)
        self._finalize_load(self._pipe)
        logger.info("Model ready.")
