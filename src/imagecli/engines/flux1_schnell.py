"""FLUX.1-schnell engine — Apache 2.0, ungated, fast generation at ~10GB VRAM (fp8)."""

from __future__ import annotations

from imagecli.engine import ImageEngine


class Flux1SchnellEngine(ImageEngine):
    name = "flux1-schnell"
    description = (
        "FLUX.1-schnell fp8 — Apache 2.0, fast 4-step generation, ~10GB VRAM (Black Forest Labs)"
    )
    model_id = "black-forest-labs/FLUX.1-schnell"
    vram_gb = 10.0

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline

        print(f"Loading {self.model_id} (fp8) …")
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # fp8 quantize the transformer to fit in 16GB cards
        try:
            from optimum.quanto import freeze, qfloat8, quantize  # type: ignore[import-untyped]

            quantize(self._pipe.transformer, weights=qfloat8)
            freeze(self._pipe.transformer)
            print("Transformer quantized to fp8.")
        except Exception as e:
            print(f"Warning: fp8 quantization failed ({e}), using bf16.")

        self._finalize_load(self._pipe)
        print("Model ready.")

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
