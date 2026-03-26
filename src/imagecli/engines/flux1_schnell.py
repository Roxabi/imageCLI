"""FLUX.1-schnell engine — GGUF Q5_K_S, Apache 2.0, fast 4-step generation at ~10GB VRAM."""

from __future__ import annotations

import logging

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)

GGUF_REPO = "city96/FLUX.1-schnell-gguf"
GGUF_FILE = "flux1-schnell-Q5_K_S.gguf"


class Flux1SchnellEngine(ImageEngine):
    name = "flux1-schnell"
    description = (
        "FLUX.1-schnell GGUF Q5_K_S — Apache 2.0, fast 4-step generation, "
        "~10GB VRAM (Black Forest Labs)"
    )
    model_id = "black-forest-labs/FLUX.1-schnell"
    vram_gb = 10.0
    # steps are NOT fixed (user can override the 4-step default)
    capabilities = EngineCapabilities(negative_prompt=False, fixed_guidance=0.0)

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
        # Sequential offload: only the active component lives on GPU at any
        # given time. The GGUF transformer is ~8 GB, but T5 (5B bf16) + CLIP +
        # VAE together exceed 16 GB if loaded simultaneously.
        self._pipe.enable_model_cpu_offload()
        self._optimize_pipe(self._pipe)
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
