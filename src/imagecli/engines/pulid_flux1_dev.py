"""FLUX.1-dev GGUF Q5_K_S + PuLID face identity lock engine.

Architecture:
- Transformer: FLUX.1-dev loaded as GGUF Q5_K_S (~6 GB on GPU)
- PuLID: injects identity into 19 double blocks (every 2nd) + 38 single blocks (every 4th)
- CA count: ceil(19/2) + ceil(38/4) = 10 + 10 = 20 modules

Face reference image path must be supplied via ``face_image`` in frontmatter.
``pulid_strength`` controls identity lock intensity (default 0.8).

VRAM profile (with enable_model_cpu_offload):
- GGUF transformer: ~6 GB on GPU
- PuLID CA + IDFormer: ~0.2 GB (stays on CUDA permanently)
- EVA-CLIP: ~0.8 GB (moved to CPU after id extraction)
- Peak during forward: ~8-10 GB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from imagecli.engine import EngineCapabilities, ImageEngine
from imagecli.engines._pulid import PuLIDFlux1, patch_flux1

logger = logging.getLogger(__name__)

_PULID_WEIGHTS = Path.home() / "ComfyUI/models/pulid/pulid_flux_v0.9.1.safetensors"
_INSIGHTFACE_DIR = Path.home() / "ComfyUI/models/insightface"

# EVA-CLIP intermediate block indices (5 evenly-spaced layers across 24 ViT blocks)
_EVA_INTERMEDIATE_INDICES = [4, 8, 12, 16, 20]

GGUF_REPO = "city96/FLUX.1-dev-gguf"
GGUF_FILE = "flux1-dev-Q5_K_S.gguf"


class PuLIDFlux1DevEngine(ImageEngine):
    name = "pulid-flux1-dev"
    description = "FLUX.1-dev GGUF Q5_K_S + PuLID face lock — requires face_image in frontmatter"
    model_id = "black-forest-labs/FLUX.1-dev"
    vram_gb = 8.0  # peak ~8-10 GB with cpu_offload
    capabilities = EngineCapabilities(negative_prompt=False)

    def __init__(self, *, compile: bool = True) -> None:
        # torch.compile captures original forward methods — incompatible with per-generation patching
        super().__init__(compile=False)
        self._pulid: PuLIDFlux1 | None = None
        self._insightface: object | None = None
        self._eva_clip_trunk: object | None = None
        self._eva_clip_head: object | None = None
        self._cached_face_path: str | None = None
        self._cached_id_tokens: torch.Tensor | None = None

    def _load(self) -> None:
        if self._pipe is not None:
            return

        import os

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

        logger.info("Loading FLUX.1-dev GGUF %s for PuLID…", GGUF_FILE)
        transformer = FluxTransformer2DModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/blob/main/{GGUF_FILE}",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        self._pipe.enable_model_cpu_offload()
        self._optimize_pipe(self._pipe)

        logger.info("Loading PuLID weights from %s…", _PULID_WEIGHTS)
        self._pulid = PuLIDFlux1.from_safetensors(_PULID_WEIGHTS)
        self._pulid.eval().to("cuda", dtype=torch.bfloat16)

        logger.info("Loading InsightFace (AntelopeV2)…")
        from insightface.app import FaceAnalysis  # type: ignore[import-untyped]

        self._insightface = FaceAnalysis(
            name="antelopev2",
            root=str(_INSIGHTFACE_DIR),
            providers=["CUDAExecutionProvider"],
        )
        self._insightface.prepare(ctx_id=0, det_size=(640, 640))  # type: ignore[union-attr]

        logger.info("Loading EVA-CLIP (EVA02-L-14-336)…")
        import open_clip  # type: ignore[import-untyped]

        clip_model, _, _ = open_clip.create_model_and_transforms(
            "EVA02-L-14-336",
            pretrained="merged2b_s6b_b61k",
        )
        # Keep trunk and head separate so we can extract both CLS and hidden features
        self._eva_clip_trunk = clip_model.visual.trunk.eval().to("cuda")
        self._eva_clip_head = clip_model.visual.trunk.head.eval().to("cuda")

        logger.info("PuLID-FLUX.1-dev engine ready.")

    def _extract_id_tokens(self, face_image_path: str) -> torch.Tensor:
        """Extract 32 identity tokens (B=1, 32, 2048) from a reference face image."""
        from PIL import Image

        device = torch.device("cuda")
        dtype = torch.bfloat16

        img_np = np.array(Image.open(face_image_path).convert("RGB"))
        faces = self._insightface.get(img_np)  # type: ignore[union-attr]
        if not faces:
            raise RuntimeError(f"No face detected in reference image: {face_image_path}")

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # InsightFace 512-d ArcFace embedding
        id_ante = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
        id_ante = F.normalize(id_ante, dim=-1)  # (1, 512)

        # Crop face region for EVA-CLIP
        x1, y1, x2, y2 = face.bbox.astype(int)
        margin = int(max(x2 - x1, y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(img_np.shape[1], x2 + margin), min(img_np.shape[0], y2 + margin)

        face_t = (
            torch.from_numpy(img_np[y1:y2, x1:x2].astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        face_t = F.interpolate(face_t, size=(336, 336), mode="bilinear", align_corners=False)

        # Normalize with EVA-CLIP stats (matches open_clip defaults for this model)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        face_t = (face_t - mean) / std

        with torch.no_grad():
            # forward_intermediates returns (final_output, list[intermediates])
            # final_output: (1, 577, 1024) where [0] is CLS token
            # intermediates: list of 5 x (1, 576, 1024) patch feature maps
            trunk_out, id_vit_hidden = self._eva_clip_trunk.forward_intermediates(  # type: ignore[union-attr]
                face_t.float(),
                indices=_EVA_INTERMEDIATE_INDICES,
                return_prefix_tokens=False,
                output_fmt="NLC",
                norm=True,
                intermediates_only=False,
            )
            # CLS token -> project to 768-d via head, then normalize
            cls_raw = trunk_out[:, 0, :]  # (1, 1024)
            id_cond_vit = self._eva_clip_head(cls_raw).to(dtype=dtype)  # type: ignore[union-attr]  # (1, 768)
            id_cond_vit = F.normalize(id_cond_vit, p=2, dim=-1)

            # id_vit_hidden: each (1, 576, 1024) — cast to bfloat16 for IDFormer
            id_vit_hidden = [f.to(dtype=dtype) for f in id_vit_hidden]

            # Concatenate ArcFace + EVA CLS -> (1, 512+768=1280) input to IDFormer
            id_cond = torch.cat([id_ante, id_cond_vit], dim=-1)  # (1, 1280)

            id_tokens = self._pulid.pulid_encoder(id_cond, id_vit_hidden)  # type: ignore[union-attr]
            id_tokens = F.normalize(id_tokens, p=2, dim=-1)  # (1, 32, 2048)

        return id_tokens

    def generate(
        self,
        prompt: str,
        *,
        face_image: str | None = None,
        pulid_strength: float = 0.8,
        output_path: Path,
        **kwargs,
    ) -> Path:
        import gc

        if not face_image:
            raise ValueError(
                "pulid-flux1-dev requires 'face_image' in the prompt frontmatter.\n"
                "Example:\n  face_image: /path/to/reference.png"
            )

        self._load()

        # Cache id_tokens — same face image = same tokens, no need to re-extract
        if self._cached_face_path != face_image:
            id_tokens = self._extract_id_tokens(face_image)
            self._cached_id_tokens = id_tokens
            self._cached_face_path = face_image
            # EVA-CLIP is done — move to CPU permanently (only needed for extraction)
            if self._eva_clip_trunk is not None:
                self._eva_clip_trunk.to("cpu")  # type: ignore[union-attr]
            if self._eva_clip_head is not None:
                self._eva_clip_head.to("cpu")  # type: ignore[union-attr]
            gc.collect()
            torch.cuda.empty_cache()

        assert self._pipe is not None
        assert self._pulid is not None
        assert self._cached_id_tokens is not None
        from typing import Any, cast as _cast
        pipe: Any = self._pipe
        unpatch: Callable[[], None] = _cast(
            Callable[[], None],
            patch_flux1(
                pipe.transformer, self._pulid, self._cached_id_tokens, pulid_strength
            ),
        )
        try:
            return super().generate(prompt, output_path=output_path, **kwargs)
        finally:
            unpatch()
            gc.collect()
            torch.cuda.empty_cache()

    def cleanup(self) -> None:
        self._pulid = None
        self._insightface = None
        self._eva_clip_trunk = None
        self._eva_clip_head = None
        self._cached_face_path = None
        self._cached_id_tokens = None
        super().cleanup()
