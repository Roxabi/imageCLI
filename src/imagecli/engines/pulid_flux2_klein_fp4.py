"""FLUX.2-klein-4B + PuLID face identity lock, NVFP4-quantized — Blackwell only.

Combines the NVFP4 runtime quantization path from flux2_klein_fp4 with the
PuLID CA activation injection from pulid_flux2_klein.

Load path:
  1. Load BF16 base transformer (same repo as pulid-flux2-klein)
  2. Runtime-quantize all nn.Linear to NVFP4 via QuantizedTensor.from_float
     (same function as flux2-klein-fp4 LoRA path)
  3. Place all components on GPU — no cpu_offload (NVFP4 kernels require CUDA)
  4. Override _execution_device so the pipeline routes denoising to CUDA

Generation path: identical to pulid-flux2-klein (EVA-CLIP offload → _patch_flux
→ super().generate → unpatch → EVA-CLIP restore).

Requires: sm_120+ (RTX 5070 Ti / Blackwell), CUDA 13.0+, comfy-kitchen.
Install: uv sync --extra pulid --group fp4
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from imagecli.engine import EngineCapabilities, ImageEngine
from imagecli.engines.pulid_flux2_klein import (
    _INSIGHTFACE_DIR,
    _PULID_DEFAULT,
    _PuLIDFlux2,
    _patch_flux,
)

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
        self._pulid: _PuLIDFlux2 | None = None
        self._insightface: object | None = None
        self._eva_clip: object | None = None

    # ── hardware + dependency checks ──────────────────────────────────────────

    def _check_requirements(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("pulid-flux2-klein-fp4 requires a CUDA GPU.")
        sm = torch.cuda.get_device_capability()
        if sm < (12, 0):
            raise RuntimeError(
                f"pulid-flux2-klein-fp4 requires Blackwell GPU (sm_120+), "
                f"got sm_{sm[0]}{sm[1]}."
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

        Without this, pipelines that skip enable_model_cpu_offload() may resolve
        the execution device from offload hooks that don't exist and silently
        fall back to CPU.
        """
        self._pipe._execution_device_override = torch.device("cuda")  # type: ignore[union-attr]
        orig_cls = type(self._pipe)
        if not hasattr(orig_cls, "_orig_execution_device"):
            orig_cls._orig_execution_device = orig_cls._execution_device  # type: ignore[attr-defined]
            orig_cls._execution_device = property(  # type: ignore[attr-defined]
                lambda pipe: getattr(pipe, "_execution_device_override", None)
                or orig_cls._orig_execution_device.fget(pipe)
            )

    # ── load ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._pipe is not None:
            return

        import os

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        self._check_requirements()

        from diffusers import Flux2KleinPipeline

        from imagecli.engines.flux2_klein_fp4 import _runtime_quantize_transformer_to_nvfp4

        logger.info("Loading Flux2Klein BF16 base for PuLID-FP4…")
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )

        # Quantize transformer to NVFP4 before moving other components to GPU.
        # Transformer lands first (~2 GB), then text encoder (~4.5 GB) + VAE (~0.2 GB).
        self._pipe.transformer.to("cuda")  # type: ignore[union-attr]
        n_quantized = _runtime_quantize_transformer_to_nvfp4(self._pipe.transformer)  # type: ignore[union-attr]
        logger.info("Runtime-quantized %d linear layers to NVFP4.", n_quantized)

        self._pipe.text_encoder.to("cuda")  # type: ignore[union-attr]
        self._pipe.vae.to("cuda")  # type: ignore[union-attr]
        self._set_execution_device()
        self._optimize_pipe(self._pipe, compile=False)

        # PuLID model — same weights and path as pulid-flux2-klein
        logger.info("Loading PuLID model from %s…", _PULID_DEFAULT)
        self._pulid = _PuLIDFlux2.from_safetensors(_PULID_DEFAULT)
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
        self._eva_clip = clip_model.visual.eval().to("cuda")
        logger.info("PuLID-FP4 engine ready.")

    # ── identity extraction (copied from PuLIDFlux2KleinEngine) ──────────────

    def _extract_id_tokens(self, face_image_path: str) -> torch.Tensor:
        from PIL import Image

        device = torch.device("cuda")
        dtype = torch.bfloat16

        img = np.array(Image.open(face_image_path).convert("RGB"))
        faces = self._insightface.get(img)  # type: ignore[union-attr]
        if not faces:
            raise RuntimeError(f"No face detected in reference image: {face_image_path}")

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        id_embed = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
        id_embed = F.normalize(id_embed, dim=-1)

        x1, y1, x2, y2 = face.bbox.astype(int)
        margin = int(max(x2 - x1, y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(img.shape[1], x2 + margin), min(img.shape[0], y2 + margin)

        face_t = (
            torch.from_numpy(img[y1:y2, x1:x2].astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        face_t = F.interpolate(face_t, size=(336, 336), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        face_t = (face_t - mean) / std

        with torch.no_grad():
            clip_out = self._eva_clip(face_t.float())  # type: ignore[union-attr]
            if isinstance(clip_out, (list, tuple)):
                clip_out = clip_out[0]
            if clip_out.dim() == 3:
                clip_out = clip_out[:, 0, :]
            clip_embed = clip_out.to(device, dtype=dtype)

            id_tokens = self._pulid.id_former(id_embed, clip_embed)  # type: ignore[union-attr]
            id_tokens = F.normalize(id_tokens, p=2, dim=-1)

        return id_tokens

    # ── generate ──────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        face_image: str | None = None,
        pulid_strength: float = 0.6,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        if not face_image:
            raise ValueError(
                "pulid-flux2-klein-fp4 requires 'face_image' in the prompt frontmatter.\n"
                "Example: face_image: /path/to/reference.png"
            )

        self._load()
        id_tokens = self._extract_id_tokens(face_image)

        # EVA-CLIP is done — move to CPU to reclaim ~1 GB before denoising
        if self._eva_clip is not None:
            self._eva_clip.to("cpu")
        torch.cuda.empty_cache()

        unpatch = _patch_flux(self._pipe.transformer, self._pulid, id_tokens, pulid_strength)  # type: ignore[union-attr]
        try:
            return super().generate(prompt, output_path=output_path, **kwargs)
        finally:
            unpatch()
            if self._eva_clip is not None:
                self._eva_clip.to("cuda")

    # ── cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        self._pulid = None
        self._insightface = None
        self._eva_clip = None
        super().cleanup()
