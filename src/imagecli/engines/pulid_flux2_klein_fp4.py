"""FLUX.2-klein-4B + PuLID face identity lock, NVFP4-quantized — Blackwell only.

Combines the NVFP4 runtime quantization path from nvfp4_quantize with the
PuLID CA activation injection from pulid_flux2_klein.

Load path:
  1. Load BF16 base transformer (same repo as pulid-flux2-klein)
  2. Runtime-quantize all nn.Linear to NVFP4 via QuantizedTensor.from_float
     (same function as flux2-klein-fp4 LoRA path)
  3. Place all components on GPU — no cpu_offload (NVFP4 kernels require CUDA)
  4. Override _execution_device so the pipeline routes denoising to CUDA

Generation path: identical to pulid-flux2-klein (EVA-CLIP offload → patch_flux2
→ super().generate → unpatch → EVA-CLIP restore).

Requires: sm_120+ (RTX 5070 Ti / Blackwell), CUDA 13.0+, comfy-kitchen.
Install: uv sync --extra pulid --group fp4
"""

from __future__ import annotations

import logging
from pathlib import Path

from imagecli.engine import EngineCapabilities, ImageEngine
from imagecli.engines._pulid import PuLIDFlux2, extract_id_tokens, patch_flux2
from imagecli.engines.pulid_flux2_klein import _INSIGHTFACE_DIR, _PULID_DEFAULT

logger = logging.getLogger(__name__)


class PuLIDFlux2KleinFP4Engine(ImageEngine):
    name = "pulid-flux2-klein-fp4"
    description = (
        "FLUX.2-klein-4B + PuLID face lock, NVFP4 quantized — Blackwell only "
        "(sm_120+, ~8.5 GB). Requires face_image in frontmatter."
    )
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    vram_gb = 8.5
    capabilities = EngineCapabilities(negative_prompt=False)

    def __init__(self, *, compile: bool = True, **kwargs: object) -> None:
        # torch.compile captures original forward methods — incompatible with per-generation
        # PuLID patching. Force compile=False, same as pulid-flux2-klein.
        # LoRA not supported; absorb kwargs to allow batch() passthrough.
        super().__init__(compile=False, **kwargs)  # type: ignore[arg-type]
        self._pulid: PuLIDFlux2 | None = None
        self._insightface: object | None = None
        self._eva_clip: object | None = None

    # ── hardware + dependency checks ──────────────────────────────────────────

    def _check_requirements(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("pulid-flux2-klein-fp4 requires a CUDA GPU.")
        sm = torch.cuda.get_device_capability()
        if sm < (12, 0):
            raise RuntimeError(
                f"pulid-flux2-klein-fp4 requires Blackwell GPU (sm_120+), got sm_{sm[0]}{sm[1]}."
            )
        cuda_ver = torch.version.cuda
        if cuda_ver and int(cuda_ver.split(".")[0]) < 13:
            raise RuntimeError(
                f"pulid-flux2-klein-fp4 requires CUDA 13.0+ (cu130), got {cuda_ver}."
            )
        try:
            import comfy_kitchen  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "pulid-flux2-klein-fp4 requires comfy-kitchen. "
                "Install: uv sync --extra pulid --group fp4"
            )

    # ── execution device override (same logic as Flux2KleinFP4Engine) ─────────

    def _set_execution_device(self) -> None:
        """Override pipeline._execution_device so all-on-GPU denoising routes to CUDA.

        Creates a per-instance anonymous subclass with a fixed _execution_device
        property so the patch is isolated to this pipeline object and is fully
        reversed in cleanup() by restoring the original class.
        """
        import torch  # noqa: F401 (torch.device used below)

        orig_cls = type(self._pipe)
        patched_cls = type(
            f"_{orig_cls.__name__}AllGPU",
            (orig_cls,),
            {"_execution_device": property(lambda _self: torch.device("cuda"))},
        )
        self._pipe.__class__ = patched_cls  # type: ignore[union-attr]

    # ── load ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._pipe is not None:
            return

        import os

        import torch

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        self._check_requirements()

        from imagecli.engine import preflight_check

        preflight_check(self)

        from diffusers import Flux2KleinPipeline

        from imagecli.engines.nvfp4_quantize import runtime_quantize_transformer_to_nvfp4

        logger.info("Loading Flux2Klein BF16 base for PuLID-FP4…")
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )

        # Runtime-quantize transformer to NVFP4 — weights are moved to CUDA
        # layer-by-layer inside runtime_quantize_transformer_to_nvfp4, so no
        # explicit transformer.to("cuda") is needed before quantization. This
        # avoids a ~10 GB BF16 peak that would occur if the full transformer
        # were moved to GPU before quantization begins.
        n_quantized = runtime_quantize_transformer_to_nvfp4(self._pipe.transformer)  # type: ignore[union-attr]
        logger.info("Runtime-quantized %d linear layers to NVFP4.", n_quantized)

        # Move remaining non-linear params (norms, biases, embeddings) to CUDA.
        # NVFP4 linear weights are already on CUDA after quantization; this call
        # only touches the residual ~1 GB BF16 params that stayed on CPU.
        self._pipe.transformer.to("cuda")  # type: ignore[union-attr]

        # Qwen3 text encoder stays on CPU — moved to CUDA only for the
        # prompt-encoding step inside generate(), then offloaded back.  Keeping it
        # on CUDA permanently would leave only ~3.5 GB headroom, which is not enough
        # for 1024×1024 denoising + PuLID CA modules.
        self._pipe.vae.to("cuda")  # type: ignore[union-attr]
        self._set_execution_device()
        self._optimize_pipe(self._pipe, compile=False)

        # PuLID model — same weights and path as pulid-flux2-klein
        logger.info("Loading PuLID model from %s…", _PULID_DEFAULT)
        self._pulid = PuLIDFlux2.from_safetensors(_PULID_DEFAULT)
        self._pulid.eval().to("cuda", dtype=torch.bfloat16)

        logger.info("Loading InsightFace (AntelopeV2)…")
        from insightface.app import FaceAnalysis  # type: ignore[import-untyped]

        self._insightface = FaceAnalysis(
            name="antelopev2",
            root=str(_INSIGHTFACE_DIR),
            providers=["CUDAExecutionProvider"],
        )
        self._insightface.prepare(ctx_id=0, det_size=(640, 640))  # type: ignore[union-attr]

        logger.info("Loading EVA-CLIP…")
        import open_clip  # type: ignore[import-untyped]

        clip_model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k",
        )
        # EVA-CLIP stays on CPU at rest — generate() moves it to CUDA only during
        # face embedding extraction, then back to CPU.
        self._eva_clip = clip_model.visual.eval()
        logger.info("PuLID-FP4 engine ready.")

    # ── generate ──────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        face_image: str | None = None,
        face_images: list[str] | None = None,
        pulid_strength: float = 0.6,
        output_path: Path,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        guidance: float = 3.5,
        seed: int | None = None,
        callback: object = None,
        **kwargs: object,
    ) -> Path:
        refs = face_images or ([face_image] if face_image else None)
        if not refs:
            raise ValueError(
                "pulid-flux2-klein-fp4 requires 'face_image' or 'face_images' in the prompt frontmatter.\n"
                "Example: face_image: /path/to/reference.png\n"
                "         face_images: [/path/to/ref1.png, /path/to/ref2.png]"
            )

        import gc
        import random

        import torch

        self._load()

        # ── Step 1: face id tokens (EVA-CLIP temporarily on CUDA) ─────────────
        if self._eva_clip is not None:
            self._eva_clip.to("cuda")  # type: ignore[union-attr]
        id_tokens = extract_id_tokens(self._insightface, self._eva_clip, self._pulid, refs)
        if self._eva_clip is not None:
            self._eva_clip.to("cpu")  # type: ignore[union-attr]
        torch.cuda.empty_cache()
        gc.collect()

        # ── Step 2: prompt encoding (Qwen3 encoder temporarily on CUDA) ─────────
        # Qwen3 text encoder is kept on CPU at rest; moved to CUDA here and back
        # to cap peak VRAM during encoding at NVFP4 + Qwen3 + PuLID ≤ 12 GB.
        self._pipe.text_encoder.to("cuda")  # type: ignore[union-attr]
        try:
            with torch.inference_mode():
                # encode_prompt returns (prompt_embeds [B,S,D], text_ids [B,S,4])
                prompt_embeds, _text_ids = self._pipe.encode_prompt(  # type: ignore[union-attr]
                    prompt=prompt,
                    device=torch.device("cuda"),
                    num_images_per_prompt=1,
                )
        finally:
            self._pipe.text_encoder.to("cpu")  # type: ignore[union-attr]
            torch.cuda.empty_cache()
            gc.collect()

        # ── Step 3: denoising with PuLID CA injection ─────────────────────────
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pipe_kwargs: dict = {
            "prompt_embeds": prompt_embeds,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if callback is not None:
            pipe_kwargs["callback_on_step_end"] = callback

        unpatch = patch_flux2(self._pipe.transformer, self._pulid, id_tokens, pulid_strength)  # type: ignore[union-attr]
        try:
            with torch.inference_mode():
                result = self._pipe(**pipe_kwargs)  # type: ignore[union-attr]
        finally:
            unpatch()

        return self._save_image(
            result.images[0],
            output_path,
            seed=seed,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
        )

    # ── cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        # Restore the original pipeline class before teardown so the per-instance
        # AllGPU subclass created by _set_execution_device() doesn't leak.
        if self._pipe is not None and type(self._pipe).__name__.endswith("AllGPU"):
            self._pipe.__class__ = type(self._pipe).__bases__[0]
        self._pulid = None
        self._insightface = None
        self._eva_clip = None
        super().cleanup()
