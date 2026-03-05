"""FLUX.1-dev engine — fp8 quantized, excellent quality at ~10GB VRAM."""

from __future__ import annotations

from imagecli.engine import ImageEngine


class Flux1DevEngine(ImageEngine):
    name = "flux1-dev"
    description = "FLUX.1-dev fp8 — top quality, ~10GB VRAM with quantization (Black Forest Labs)"
    model_id = "black-forest-labs/FLUX.1-dev"
    vram_gb = 10.0

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline  # type: ignore[import-untyped]

        print(f"Loading {self.model_id} (fp8) …")
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # fp8 quantize the transformer to save VRAM
        try:
            from optimum.quanto import freeze, qfloat8, quantize  # type: ignore[import-untyped]

            quantize(self._pipe.transformer, weights=qfloat8)
            freeze(self._pipe.transformer)
            print("Transformer quantized to fp8.")
        except Exception as e:
            print(f"Warning: fp8 quantization failed ({e}), using bf16.")

        self._finalize_load(self._pipe)
        print("Model ready.")
