"""FLUX.1-schnell engine — Apache 2.0, ungated, fast 4-step generation at ~10GB VRAM."""

from __future__ import annotations

from imagecli.engine import ImageEngine, _get_compute_capability


class Flux1SchnellEngine(ImageEngine):
    name = "flux1-schnell"
    description = "FLUX.1-schnell quantized — Apache 2.0, fast 4-step generation, ~10GB VRAM (Black Forest Labs)"
    model_id = "black-forest-labs/FLUX.1-schnell"
    vram_gb = 10.0

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline

        sm = _get_compute_capability()
        qtype_label = "fp8" if sm >= (8, 9) else "int8"
        print(f"Loading {self.model_id} ({qtype_label}) …")
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # fp8 on Ada Lovelace/Blackwell (sm≥89); int8 on Ampere (sm≥80) — both ~10GB
        try:
            from optimum.quanto import freeze, quantize  # type: ignore[import-untyped]

            if sm >= (8, 9):
                from optimum.quanto import qfloat8  # type: ignore[import-untyped]

                quantize(self._pipe.transformer, weights=qfloat8)
            else:
                from optimum.quanto import qint8  # type: ignore[import-untyped]

                quantize(self._pipe.transformer, weights=qint8)
            freeze(self._pipe.transformer)
            print(f"Transformer quantized to {qtype_label}.")
        except Exception as e:
            print(f"Warning: quantization failed ({e}), using bf16.")

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
