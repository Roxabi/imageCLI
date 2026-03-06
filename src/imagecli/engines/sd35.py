"""SD3.5 Large Turbo engine — fast, 20-step, fits in 16GB with quantization."""

from __future__ import annotations

from imagecli.engine import EngineCapabilities, ImageEngine


class SD35Engine(ImageEngine):
    name = "sd35"
    description = "Stable Diffusion 3.5 Large Turbo — fast 20-step generation (Stability AI)"
    model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
    vram_gb = 14.0
    # negative_prompt=True stays as default (sd35 supports it)
    capabilities = EngineCapabilities(fixed_steps=20, fixed_guidance=1.0)

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableDiffusion3Pipeline  # type: ignore[import-untyped]

        print(f"Loading {self.model_id} …")
        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # quantize T5 text encoder (biggest VRAM hog in SD3.5) to fit in 16GB
        try:
            from optimum.quanto import freeze, qint8, quantize  # type: ignore[import-untyped]

            quantize(self._pipe.text_encoder_3, weights=qint8)
            freeze(self._pipe.text_encoder_3)
            print("T5 encoder quantized to int8.")
        except Exception as e:
            print(f"Warning: T5 quantization failed ({e}), using bf16.")

        self._finalize_load(self._pipe)
        print("Model ready.")

    def effective_steps(self, requested: int) -> int:
        # Turbo always runs exactly 20 steps regardless of what the caller requested.
        return 20

    def _build_pipe_kwargs(
        self, prompt, *, negative_prompt, width, height, steps, guidance, generator
    ):
        # Turbo: always 20 steps, guidance 1.0 (CFG-free)
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "width": width,
            "height": height,
            "num_inference_steps": 20,
            "guidance_scale": 1.0,
            "generator": generator,
        }
