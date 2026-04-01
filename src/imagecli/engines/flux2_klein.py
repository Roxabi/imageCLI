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
        import gc

        import torch
        from diffusers import Flux2KleinPipeline
        from optimum.quanto import freeze, qfloat8, quantize

        logger.info("Loading %s...", self.model_id)
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # Quantize transformer to FP8: 7.75 GB → ~3.9 GB.
        # Total VRAM: text_encoder 7.49 + transformer 3.9 + VAE 0.17 = ~11.6 GB.
        # Fits in 15.45 GB without offload.
        logger.info("Quantizing transformer to float8...")
        quantize(self._pipe.transformer, weights=qfloat8)
        freeze(self._pipe.transformer)
        # Marlin FP8 GEMM kernel requires contiguous input
        from optimum.quanto.nn import QLinear
        _orig_fwd = QLinear.forward
        def _fwd_cont(self, input):
            return _orig_fwd(self, input.contiguous())
        QLinear.forward = _fwd_cont
        # CPU offload: text encoder (7.5 GB) must be off GPU before VAE decode
        # at 1024x1024. Offload moves each component to GPU only when needed.
        self._pipe.enable_model_cpu_offload()
        self._optimize_pipe(self._pipe)
        logger.info("Model ready.")
