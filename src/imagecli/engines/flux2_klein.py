"""FLUX.2-klein-4B engine — default, FP8 quantized.

Single generate: CPU offload (~8 GB peak, no compile, fastest for 1-2 images).
Batch: 2-phase (encode all prompts → generate all images, ~4 GB peak in phase 2, compiled).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)


class Flux2KleinEngine(ImageEngine):
    name = "flux2-klein"
    description = "FLUX.2-klein-4B — best quality for 16GB VRAM (Black Forest Labs, Nov 2025)"
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    vram_gb = 8.0  # FP8 + CPU offload, peak ~7.84 GB
    capabilities = EngineCapabilities(negative_prompt=False)
    supports_two_phase: ClassVar[bool] = True

    def _load_pipeline(self):
        """Shared setup: load from pretrained, quantize transformer to FP8."""
        if self._pipe is not None:
            return

        import torch
        from diffusers import Flux2KleinPipeline
        from optimum.quanto import freeze, qfloat8, quantize

        logger.info("Loading %s...", self.model_id)
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        # LoRA must be loaded BEFORE quantization — weights are fused into bf16 base,
        # then quantized together. Loading after quantization silently has no effect.
        if self.loras:
            adapter_names = []
            for i, spec in enumerate(self.loras):
                name = f"lora_{i}"
                logger.info("Loading LoRA %d/%d from %s...", i + 1, len(self.loras), spec.path)
                self._pipe.load_lora_weights(spec.path, adapter_name=name)
                adapter_names.append(name)
            if len(adapter_names) > 1:
                # N≥2: adapters have distinct weights → set_adapters carries
                # per-adapter scale, then fuse_lora(1.0) applies no global
                # multiplier on top.
                self._pipe.set_adapters(
                    adapter_names, adapter_weights=[s.scale for s in self.loras]
                )
                logger.info(
                    "Set %d adapters with scales %s.",
                    len(adapter_names),
                    [s.scale for s in self.loras],
                )
            # N=1: diffusers' fuse_lora(lora_scale=s) applies a global scalar
            # multiplier across the only active adapter — equivalent to
            # set_adapters(["lora_0"], [s]) + fuse_lora(1.0) for a single
            # adapter (the per-adapter weight and the global multiplier
            # combine multiplicatively, so scale*1.0 == 1.0*scale at a single
            # adapter). This is the pre-#34 single-LoRA behavior preserved.
            fuse_scale = self.loras[0].scale if len(self.loras) == 1 else 1.0
            self._pipe.fuse_lora(lora_scale=fuse_scale)
            self._pipe.unload_lora_weights()
            logger.info("LoRA(s) fused into base weights.")
        # Pivotal tuning: load trained trigger vectors into the TE BEFORE
        # transformer quantization. Touches tokenizer + embed_tokens only;
        # disjoint from LoRA fuse/unload and transformer quantize.
        self._apply_pivotal_embeddings()
        # Quantize transformer to FP8: 7.75 GB → ~3.9 GB.
        logger.info("Quantizing transformer to float8...")
        quantize(self._pipe.transformer, weights=qfloat8)
        freeze(self._pipe.transformer)
        # Marlin FP8 GEMM kernel requires contiguous input
        from optimum.quanto.nn import QLinear

        _orig_fwd = QLinear.forward

        def _fwd_cont(self, input):
            return _orig_fwd(self, input.contiguous())

        QLinear.forward = _fwd_cont

    def _load(self):
        """Single-image mode: CPU offload, no compile."""
        if self._pipe is not None:
            return
        self._load_pipeline()
        # CPU offload: layers moved to GPU on-demand, rest stays in RAM.
        # Peak VRAM: ~7.84 GB (transformer chunks + VAE). Fits in 12GB+ with headroom.
        self._pipe.enable_model_cpu_offload()
        self._optimize_pipe(self._pipe, compile=False)  # compile incompatible with offload hooks
        logger.info("Model ready (CPU offload mode).")

    # ── All-on-GPU batch ─────────────────────────────────────────────────

    def load_all_on_gpu(self):
        """Load everything to GPU at once (~12 GB). No offloading between phases."""
        self._load_pipeline()
        self._pipe.text_encoder.to("cuda")
        self._pipe.transformer.to("cuda")
        self._pipe.vae.to("cuda")
        self._pipe._execution_device_override = __import__("torch").device("cuda")
        orig_cls = type(self._pipe)
        if not hasattr(orig_cls, "_orig_execution_device"):
            orig_cls._orig_execution_device = orig_cls._execution_device
            orig_cls._execution_device = property(
                lambda pipe: (
                    getattr(pipe, "_execution_device_override", None)
                    or orig_cls._orig_execution_device.fget(pipe)
                )
            )
        self._optimize_pipe(self._pipe, compile=False)
        logger.info("All components on GPU (~12 GB) — no phase switching.")

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
            image,
            output_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
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

        # Free encoder VRAM
        self._pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        import torch

        # Transformer (~3.9 GB FP8) + VAE (~0.17 GB) → ~4.1 GB on GPU
        self._pipe.transformer.to("cuda")
        self._pipe.vae.to("cuda")
        # Pipeline.device / _execution_device derives from the first registered module
        # (text_encoder, on CPU). Monkey-patch the instance method to return cuda.
        self._pipe._execution_device_override = torch.device("cuda")
        orig_cls = type(self._pipe)
        if not hasattr(orig_cls, "_orig_execution_device"):
            orig_cls._orig_execution_device = orig_cls._execution_device
            orig_cls._execution_device = property(
                lambda pipe: (
                    getattr(pipe, "_execution_device_override", None)
                    or orig_cls._orig_execution_device.fget(pipe)
                )
            )
        # Skip compile: torch.compile conflicts with QLinear.forward contiguity patch.
        self._optimize_pipe(self._pipe, compile=False)
        logger.info("Generation phase ready (transformer + VAE on GPU, ~4 GB).")

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
            image,
            output_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
        )
