"""FLUX.2-klein-4B NVFP4 engine — Blackwell-native FP4 via comfy-kitchen.

Uses BFL official NVFP4 weights with comfy-kitchen's scaled_mm_nvfp4 kernel.
~2 GB transformer, ~11.5 GB all-on-GPU, ~6.8 it/s at 512x512 on RTX 5070 Ti.

LoRA support: when --lora is set, loads bf16 base, fuses LoRA at --lora-scale,
then runtime-quantizes the fused transformer to NVFP4 via
QuantizedTensor.from_float(...). Pre-quantized disk weights are bypassed in this
path (they encode the un-fused model). V22 validation: NVFP4 + fused LoRA scores
−0.032 vs FP8 + fused LoRA, for 8× speedup and 60% VRAM reduction.

Requires Blackwell GPU (sm_120+), CUDA 13.0 (cu130), and comfy-kitchen[cublas].

Install: uv sync --group fp4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)

NVFP4_REPO = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
NVFP4_FILENAME = "flux-2-klein-4b-nvfp4.safetensors"
BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"


class Flux2KleinFP4Engine(ImageEngine):
    name = "flux2-klein-fp4"
    description = "FLUX.2-klein-4B NVFP4 — Blackwell FP4 via comfy-kitchen (~2 GB transformer)"
    model_id = BASE_REPO
    vram_gb = 4.0
    capabilities = EngineCapabilities(negative_prompt=False)
    supports_two_phase: ClassVar[bool] = True

    def _check_requirements(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("NVFP4 requires a CUDA GPU.")
        sm = torch.cuda.get_device_capability()
        if sm < (12, 0):
            raise RuntimeError(f"NVFP4 requires Blackwell GPU (sm_120+), got sm_{sm[0]}{sm[1]}.")
        cuda_ver = torch.version.cuda
        if cuda_ver and int(cuda_ver.split(".")[0]) < 13:
            raise RuntimeError(f"NVFP4 requires CUDA 13.0+ (cu130), got {cuda_ver}.")
        try:
            import comfy_kitchen  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "flux2-klein-fp4 requires comfy-kitchen. Install: uv sync --group fp4"
            )

    def _load_pipeline(self):
        if self._pipe is not None:
            return

        if self.loras:
            raise ValueError(
                "flux2-klein-fp4 does not support LoRA (model is pre-quantized NVFP4). "
                "Use flux2-klein or flux2-klein-fp8 instead."
            )

        self._check_requirements()

        import torch
        from diffusers import Flux2KleinPipeline
        from huggingface_hub import hf_hub_download

        from imagecli.engines.nvfp4_quantize import patch_transformer_nvfp4

        logger.info("Loading base pipeline %s...", BASE_REPO)
        self._pipe = Flux2KleinPipeline.from_pretrained(BASE_REPO, torch_dtype=torch.bfloat16)

        # No LoRA: use pre-quantized BFL NVFP4 weights from the hub.
        # (LoRA + pivotal cases are rejected by the guard above — #34 excluded
        # fp4 from the multi-LoRA feature; the pre-#34 single-LoRA runtime-fuse
        # path was also dropped because the guard pre-empts any LoRA request.)
        logger.info("Downloading NVFP4 weights from %s...", NVFP4_REPO)
        nvfp4_path = hf_hub_download(repo_id=NVFP4_REPO, filename=NVFP4_FILENAME)

        self._pipe.transformer.to("cuda")
        logger.info("Patching transformer with NVFP4 QuantizedTensors...")
        patched = patch_transformer_nvfp4(self._pipe.transformer, nvfp4_path)
        logger.info("Patched %d linear layers with NVFP4 weights.", patched)

    def _set_execution_device(self):
        import torch

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

    def _load(self):
        if self._pipe is not None:
            return
        self._load_pipeline()
        self._pipe.vae.to("cuda")
        self._pipe.text_encoder.to("cuda")
        self._optimize_pipe(self._pipe, compile=False)
        self._set_execution_device()
        logger.info("Model ready (all on GPU, NVFP4 transformer).")

    # ── All-on-GPU batch ─────────────────────────────────────────────────

    def load_all_on_gpu(self):
        self._load_pipeline()
        self._pipe.text_encoder.to("cuda")
        self._pipe.vae.to("cuda")
        self._set_execution_device()
        self._optimize_pipe(self._pipe, compile=False)
        logger.info("All components on GPU — NVFP4 transformer.")

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

        return self._save_image(
            result.images[0],
            output_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
        )

    # ── 2-phase batch ──────────────────────────────────────────────────────

    def load_for_encode(self):
        self._load_pipeline()
        self._pipe.text_encoder.to("cuda")
        logger.info("Text encoder on GPU — ready for prompt encoding.")

    def encode_prompt(self, prompt: str) -> dict:
        import torch

        with torch.inference_mode():
            prompt_embeds, text_ids = self._pipe.encode_prompt(
                prompt=prompt,
                device="cuda",
                num_images_per_prompt=1,
            )
        return {"prompt_embeds": prompt_embeds.cpu(), "text_ids": text_ids.cpu()}

    def start_generation_phase(self):
        import gc
        import torch

        self._pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        self._pipe.vae.to("cuda")
        self._set_execution_device()
        self._optimize_pipe(self._pipe, compile=False)
        logger.info("Generation phase ready (NVFP4 transformer + VAE on GPU).")

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

        return self._save_image(
            result.images[0],
            output_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
        )
