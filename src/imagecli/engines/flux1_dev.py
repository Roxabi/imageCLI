"""FLUX.1-dev engine — quantized (fp8 on sm≥89, int8 on Ampere), ~10GB VRAM."""

from __future__ import annotations

from imagecli.engine import ImageEngine, _get_compute_capability


class Flux1DevEngine(ImageEngine):
    name = "flux1-dev"
    description = "FLUX.1-dev quantized — top quality, ~10GB VRAM (Black Forest Labs)"
    model_id = "black-forest-labs/FLUX.1-dev"
    vram_gb = 10.0

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import FluxPipeline  # type: ignore[import-untyped]

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
