"""SD3.5 Large Turbo engine — fast, 20-step, fits in 16GB with quantization."""

from __future__ import annotations

from pathlib import Path

from imagecli.engine import ImageEngine


class SD35Engine(ImageEngine):
    name = 'sd35'
    description = 'Stable Diffusion 3.5 Large Turbo — fast 20-step generation (Stability AI)'
    model_id = 'stabilityai/stable-diffusion-3.5-large-turbo'
    vram_gb = 14.0

    def __init__(self):
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import StableDiffusion3Pipeline  # type: ignore[import-untyped]

        print(f'Loading {self.model_id} …')
        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # quantize T5 text encoder (biggest VRAM hog in SD3.5) to fit in 16GB
        try:
            from optimum.quanto import freeze, qint8, quantize  # type: ignore[import-untyped]
            quantize(self._pipe.text_encoder_3, weights=qint8)
            freeze(self._pipe.text_encoder_3)
            print('T5 encoder quantized to int8.')
        except Exception as e:
            print(f'Warning: T5 quantization failed ({e}), using bf16.')

        self._pipe.enable_model_cpu_offload()
        print('Model ready.')

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = '',
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,  # turbo is fixed at 20
        guidance: float = 1.0,  # turbo uses guidance=1.0 (CFG-free)
        seed: int | None = None,
        output_path: Path,
        **kwargs,
    ) -> Path:
        import torch

        self._load()

        generator = torch.Generator('cuda').manual_seed(seed) if seed is not None else None

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=20,  # turbo is always 20
            guidance_scale=1.0,      # turbo is CFG-free
            generator=generator,
        )
        image = result.images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path))
        return output_path
