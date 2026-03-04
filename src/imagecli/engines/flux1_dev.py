"""FLUX.1-dev engine — fp8 quantized, excellent quality at ~10GB VRAM."""

from __future__ import annotations

from pathlib import Path

from imagecli.engine import ImageEngine


class Flux1DevEngine(ImageEngine):
    name = 'flux1-dev'
    description = 'FLUX.1-dev fp8 — top quality, ~10GB VRAM with quantization (Black Forest Labs)'
    model_id = 'black-forest-labs/FLUX.1-dev'
    vram_gb = 10.0

    def __init__(self):
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline  # type: ignore[import-untyped]

        print(f'Loading {self.model_id} (fp8) …')
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # fp8 quantize the transformer to save VRAM
        try:
            from optimum.quanto import freeze, qfloat8, quantize  # type: ignore[import-untyped]
            quantize(self._pipe.transformer, weights=qfloat8)
            freeze(self._pipe.transformer)
            print('Transformer quantized to fp8.')
        except Exception as e:
            print(f'Warning: fp8 quantization failed ({e}), using bf16.')

        self._pipe.enable_model_cpu_offload()
        print('Model ready.')

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = '',
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        guidance: float = 3.5,
        seed: int | None = None,
        output_path: Path,
        **kwargs,
    ) -> Path:
        import torch

        self._load()

        generator = torch.Generator('cuda').manual_seed(seed) if seed is not None else None

        result = self._pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        image = result.images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path))
        return output_path
