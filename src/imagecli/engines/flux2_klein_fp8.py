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
from pathlib import Path
from typing import ClassVar

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)

BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"


class Flux2KleinFP8Engine(ImageEngine):
    name = "flux2-klein-fp8"
    description = "FLUX.2-klein-4B torchao FP8 — no quanto, torch.compile compatible (slower than quanto, see notes)"
    model_id = BASE_REPO
    vram_gb = 8.0
    capabilities = EngineCapabilities(negative_prompt=False)
    supports_two_phase: ClassVar[bool] = True

    def _load_pipeline(self):
        """Load base pipeline + quantize transformer to FP8 via torchao."""
        if self._pipe is not None:
            return

        try:
            from torchao.quantization import Float8WeightOnlyConfig, quantize_
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
        if self.lora_path:
            logger.info("Loading LoRA from %s...", self.lora_path)
            self._pipe.load_lora_weights(self.lora_path)
            if self.lora_scale != 1.0:
                self._pipe.set_adapters(["default_0"], adapter_weights=[self.lora_scale])
                logger.info("LoRA adapter scale set to %.2f", self.lora_scale)
            self._pipe.fuse_lora()
            self._pipe.unload_lora_weights()
            logger.info("LoRA fused into base weights.")

        # Quantize transformer to FP8 via torchao (weight-only, no QLinear patch needed)
        logger.info("Quantizing transformer to FP8 via torchao...")
        quantize_(self._pipe.transformer, Float8WeightOnlyConfig())
        logger.info("Transformer quantized to FP8 (torchao).")

    def _load(self):
        """Single-image mode: all on GPU (~12 GB). CPU offload incompatible with torchao tensors."""
        if self._pipe is not None:
            return
        self._load_pipeline()
        self._pipe.to("cuda")
        self._optimize_pipe(self._pipe, compile=False)
        logger.info("Model ready (all on GPU, torchao FP8).")

    # ── All-on-GPU batch ─────────────────────────────────────────────────

    def _set_execution_device(self):
        """Patch pipeline execution device to cuda."""
        import torch

        self._pipe._execution_device_override = torch.device("cuda")
        orig_cls = type(self._pipe)
        if not hasattr(orig_cls, "_orig_execution_device"):
            orig_cls._orig_execution_device = orig_cls._execution_device
            orig_cls._execution_device = property(
                lambda pipe: getattr(pipe, "_execution_device_override", None)
                or orig_cls._orig_execution_device.fget(pipe)
            )

    def load_all_on_gpu(self):
        """Load everything to GPU at once. No offloading between phases."""
        self._load_pipeline()
        self._pipe.text_encoder.to("cuda")
        self._pipe.transformer.to("cuda")
        self._pipe.vae.to("cuda")
        self._set_execution_device()
        # No QLinear → compile works!
        self._optimize_pipe(self._pipe)
        logger.info("All components on GPU — torchao FP8, compile enabled.")

    def encode_and_generate(
        self,
        prompt: str,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        guidance: float = 4.0,
        seed: int | None = None,
        output_path: Path,
        callback=None,
    ) -> Path:
        """Encode + generate in one shot (all-on-GPU mode)."""
        import random

        import torch

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pipe_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if callback is not None:
            pipe_kwargs["callback_on_step_end"] = callback

        with torch.inference_mode():
            result = self._pipe(**pipe_kwargs)

        image = result.images[0]
        return self._save_image(
            image, output_path, seed=seed, steps=steps, guidance=guidance, width=width, height=height
        )

    # ── 2-phase batch ──────────────────────────────────────────────────────

    def load_for_encode(self):
        """Phase 1 setup: load pipeline, move text encoder to GPU (~8 GB)."""
        self._load_pipeline()
        self._pipe.text_encoder.to("cuda")
        logger.info("Text encoder on GPU — ready for prompt encoding.")

    def encode_prompt(self, prompt: str) -> dict:
        """Encode a single prompt. Returns embeddings dict (on CPU to free VRAM)."""
        import torch

        with torch.inference_mode():
            prompt_embeds, text_ids = self._pipe.encode_prompt(
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
        import gc

        import torch

        self._pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        self._pipe.transformer.to("cuda")
        self._pipe.vae.to("cuda")
        self._set_execution_device()
        # No QLinear → compile works in 2-phase too!
        self._optimize_pipe(self._pipe)
        logger.info("Generation phase ready (transformer + VAE on GPU, compile enabled).")

    def generate_from_embeddings(
        self,
        embeddings: dict,
        *,
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        guidance: float = 4.0,
        seed: int | None = None,
        output_path: Path,
        callback=None,
    ) -> Path:
        """Generate image from pre-computed prompt embeddings."""
        import random

        import torch

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        pipe_kwargs = {
            "prompt_embeds": embeddings["prompt_embeds"].to("cuda"),
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if callback is not None:
            pipe_kwargs["callback_on_step_end"] = callback

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            result = self._pipe(**pipe_kwargs)

        image = result.images[0]
        return self._save_image(
            image, output_path, seed=seed, steps=steps, guidance=guidance, width=width, height=height
        )
