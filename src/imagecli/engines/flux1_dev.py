"""FLUX.1-dev engine — GGUF Q5_K_S, top quality, ~10GB VRAM."""

from __future__ import annotations

import logging

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)

GGUF_REPO = "city96/FLUX.1-dev-gguf"
GGUF_FILE = "flux1-dev-Q5_K_S.gguf"


class Flux1DevEngine(ImageEngine):
    name = "flux1-dev"
    description = "FLUX.1-dev GGUF Q5_K_S — top quality, ~10GB VRAM (Black Forest Labs)"
    model_id = "black-forest-labs/FLUX.1-dev"
    vram_gb = 10.0
    capabilities = EngineCapabilities(negative_prompt=False)

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

        logger.info("Loading %s (GGUF %s)...", self.model_id, GGUF_FILE)
        transformer = FluxTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{GGUF_FILE}",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        self._pipe.enable_model_cpu_offload()
        self._optimize_pipe(self._pipe)
        logger.info("Model ready.")
