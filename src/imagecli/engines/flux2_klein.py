"""FLUX.2-klein-4B engine — default, fits in ~13GB VRAM."""

from __future__ import annotations

from pathlib import Path

from imagecli.engine import ImageEngine


class Flux2KleinEngine(ImageEngine):
    name = 'flux2-klein'
    description = 'FLUX.2-klein-4B — best quality for 16GB VRAM (Black Forest Labs, Nov 2025)'
    model_id = 'black-forest-labs/FLUX.2-klein-4B'
    vram_gb = 13.0

    def __init__(self):
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import Flux2KleinPipeline  # type: ignore[import-untyped]

        print(f'Loading {self.model_id} …')
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
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
        guidance: float = 4.0,
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
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        image = result.images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path))
        return output_path
