"""FLUX.1-schnell engine — Apache 2.0, ungated, fast 4-step generation at ~10GB VRAM."""

from __future__ import annotations

import logging

from imagecli.engine import EngineCapabilities, ImageEngine, get_compute_capability

logger = logging.getLogger(__name__)


class Flux1SchnellEngine(ImageEngine):
    name = "flux1-schnell"
    description = "FLUX.1-schnell quantized — Apache 2.0, fast 4-step generation, ~10GB VRAM (Black Forest Labs)"
    model_id = "black-forest-labs/FLUX.1-schnell"
    vram_gb = 10.0
    # steps are NOT fixed (user can override the 4-step default)
    capabilities = EngineCapabilities(negative_prompt=False, fixed_guidance=0.0)

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline

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

    def _build_pipe_kwargs(
        self, prompt, *, negative_prompt, width, height, steps, guidance, generator
    ):
        return {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }

    # Override defaults for schnell: 4 steps, 0.0 guidance
    def generate(
        self,
        prompt,
        *,
        negative_prompt="",
        width=1024,
        height=1024,
        steps=4,
        guidance=0.0,
        seed=None,
        output_path,
        **kwargs,
    ):
        return super().generate(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            output_path=output_path,
            **kwargs,
        )
