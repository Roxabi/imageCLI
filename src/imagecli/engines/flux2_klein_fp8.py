"""FLUX.2-klein-4B engine — torchao FP8 quantization (no quanto).

Loads bf16 transformer, quantizes to FP8 via torchao at load time.
No QLinear contiguity patch → torch.compile works out of the box.

NOTE: As of torchao 0.17 / torch 2.11, this engine is ~40% slower than the
quanto-based flux2-klein engine (4.3 it/s vs 7.0 it/s at 512x512). torchao
weight-only FP8 dequantizes to bf16 for each matmul, while quanto uses Marlin
FP8 GEMM kernels that compute natively in FP8. Use this engine only if you
need torch.compile compatibility or want to avoid the quanto dependency.

Requires: uv sync --group fp8 (installs torchao)
"""

from __future__ import annotations

import logging

from imagecli.engine import EngineCapabilities
from imagecli.engines._two_phase_base import TwoPhaseBase
from imagecli.engines.helpers import set_execution_device

logger = logging.getLogger(__name__)

BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"


class Flux2KleinFP8Engine(TwoPhaseBase):
    name = "flux2-klein-fp8"
    description = "FLUX.2-klein-4B torchao FP8 — no quanto, torch.compile compatible (slower than quanto, see notes)"
    model_id = BASE_REPO
    vram_gb = 8.0
    capabilities = EngineCapabilities(negative_prompt=False)

    def _load_pipeline(self):
        """Load base pipeline + quantize transformer to FP8 via torchao."""
        if self._pipe is not None:
            return

        try:
            from torchao.quantization import Float8WeightOnlyConfig, quantize_  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "flux2-klein-fp8 requires torchao. Install with: uv sync --group fp8"
            )

        import torch
        from diffusers import Flux2KleinPipeline

        logger.info("Loading %s...", BASE_REPO)
        self._pipe = Flux2KleinPipeline.from_pretrained(
            BASE_REPO,
            torch_dtype=torch.bfloat16,
        )
        # LoRA must be loaded BEFORE quantization — weights are fused into bf16 base,
        # then quantized together. Loading after quantization silently has no effect.
        if self.loras:
            adapter_names = []
            for i, spec in enumerate(self.loras):
                name = f"lora_{i}"
                logger.info("Loading LoRA %d/%d from %s...", i + 1, len(self.loras), spec.path)
                self._pipe.load_lora_weights(spec.path, adapter_name=name)  # type: ignore[attr-defined]
                adapter_names.append(name)
            if len(adapter_names) > 1:
                self._pipe.set_adapters(  # type: ignore[attr-defined]
                    adapter_names, adapter_weights=[s.scale for s in self.loras]
                )
                logger.info(
                    "Set %d adapters with scales %s.",
                    len(adapter_names),
                    [s.scale for s in self.loras],
                )
            fuse_scale = self.loras[0].scale if len(self.loras) == 1 else 1.0
            self._pipe.fuse_lora(lora_scale=fuse_scale)  # type: ignore[attr-defined]
            self._pipe.unload_lora_weights()  # type: ignore[attr-defined]
            logger.info("LoRA(s) fused into base weights.")

        # Pivotal tuning: load trained trigger vectors into the TE BEFORE
        # transformer quantization. TE stays bf16 in torchao's weight-only FP8
        # quantization — only nn.Linear layers are touched.
        self._apply_pivotal_embeddings()

        # Quantize transformer to FP8 via torchao (weight-only, no QLinear patch needed)
        logger.info("Quantizing transformer to FP8 via torchao...")
        quantize_(self._pipe.transformer, Float8WeightOnlyConfig())  # type: ignore[attr-defined]
        logger.info("Transformer quantized to FP8 (torchao).")

    def _load(self):
        """Single-image mode: all on GPU (~12 GB). CPU offload incompatible with torchao tensors."""
        if self._pipe is not None:
            return
        self._load_pipeline()
        assert self._pipe is not None
        self._pipe.to("cuda")  # type: ignore[attr-defined]
        self._optimize_pipe(self._pipe, compile=False)
        logger.info("Model ready (all on GPU, torchao FP8).")

    # ── All-on-GPU batch ─────────────────────────────────────────────────

    def load_all_on_gpu(self):
        """Load everything to GPU at once. No offloading between phases."""
        self._load_pipeline()
        assert self._pipe is not None
        self._pipe.text_encoder.to("cuda")  # type: ignore[attr-defined]
        self._pipe.transformer.to("cuda")  # type: ignore[attr-defined]
        self._pipe.vae.to("cuda")  # type: ignore[attr-defined]
        set_execution_device(self._pipe)
        # No QLinear → compile works!
        self._optimize_pipe(self._pipe)
        logger.info("All components on GPU — torchao FP8, compile enabled.")

    # ── 2-phase batch ──────────────────────────────────────────────────────

    def load_for_encode(self):
        """Phase 1 setup: load pipeline, move text encoder to GPU (~8 GB)."""
        self._load_pipeline()
        assert self._pipe is not None
        self._pipe.text_encoder.to("cuda")  # type: ignore[attr-defined]
        logger.info("Text encoder on GPU — ready for prompt encoding.")

    def encode_prompt(self, prompt: str) -> dict:
        """Encode a single prompt. Returns embeddings dict (on CPU to free VRAM)."""
        import torch

        assert self._pipe is not None
        with torch.inference_mode():
            prompt_embeds, text_ids = self._pipe.encode_prompt(  # type: ignore[attr-defined]
                prompt=prompt,
                device="cuda",
                num_images_per_prompt=1,
            )
        return {
            "prompt_embeds": prompt_embeds.cpu(),
            "text_ids": text_ids.cpu(),
        }

    def start_generation_phase(self):
        """Phase 2 setup: offload encoder, load transformer + VAE to GPU, compile."""
        assert self._pipe is not None
        self._teardown_encoder_phase()

        self._pipe.transformer.to("cuda")  # type: ignore[attr-defined]
        self._pipe.vae.to("cuda")  # type: ignore[attr-defined]
        set_execution_device(self._pipe)
        # No QLinear → compile works in 2-phase too!
        self._optimize_pipe(self._pipe)
        logger.info("Generation phase ready (transformer + VAE on GPU, compile enabled).")
