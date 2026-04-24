"""FLUX.2-klein-4B + PuLID face identity lock engine.

Ported from iFayens/ComfyUI-PuLID-Flux2 — ComfyUI dependencies stripped,
runs directly in the imagecli diffusers pipeline.

Face reference image path must be supplied via ``face_image`` in frontmatter.

PuLID nn.Modules live in ``_pulid.klein_modules``; transformer patching +
Strategy B dim-projection wiring live in ``_pulid.klein_patching``;
identity-token preprocessing lives in ``_pulid.klein_preprocessing``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from imagecli.engine import EngineCapabilities, ImageEngine
from imagecli.engines._pulid import PuLIDFlux2, extract_id_tokens, patch_flux2

logger = logging.getLogger(__name__)

_PULID_DIR = Path.home() / "ComfyUI/models/pulid"
_INSIGHTFACE_DIR = Path.home() / "ComfyUI/models/insightface"
_PULID_DEFAULT = _PULID_DIR / "pulid_flux2_klein_v2.safetensors"


class PuLIDFlux2KleinEngine(ImageEngine):
    name = "pulid-flux2-klein"
    description = "FLUX.2-klein-4B + PuLID face lock — requires face_image in frontmatter"
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    vram_gb = 8.0  # with cpu_offload, peak during transformer forward ~9-10 GB
    capabilities = EngineCapabilities(negative_prompt=False)

    def __init__(self, *, compile: bool = True, **kwargs: object) -> None:
        # torch.compile captures original forward methods — incompatible with per-generation patching
        # LoRA not supported; absorb kwargs to allow get_engine() passthrough.
        super().__init__(compile=False, **kwargs)  # type: ignore[arg-type]
        self._pulid: PuLIDFlux2 | None = None
        self._insightface: object | None = None
        self._eva_clip: object | None = None

    def _finalize_load(self, pipe: object) -> None:
        """Use CPU offloading instead of full VRAM load — keeps peak ~9-10 GB."""
        import torch

        if torch.cuda.is_available():
            free_vram_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
            if free_vram_gb < self.vram_gb:
                from imagecli.engine import InsufficientResourcesError

                raise InsufficientResourcesError(
                    f"Engine {self.name!r} needs ~{self.vram_gb:.1f} GB VRAM free, "
                    f"but only {free_vram_gb:.1f} GB is available."
                )
            pipe.enable_model_cpu_offload()  # type: ignore[union-attr]
        self._optimize_pipe(pipe)

    def _load(self) -> None:
        if self._pipe is not None:
            return

        import os

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        from diffusers import Flux2KleinPipeline

        logger.info("Loading Flux2Klein for PuLID (cpu_offload)…")
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self._finalize_load(self._pipe)

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
        self._eva_clip = clip_model.visual.eval().to("cuda")
        logger.info("PuLID engine ready.")

    def _extract_id_tokens(self, face_image_paths: str | list[str]) -> torch.Tensor:
        return extract_id_tokens(self._insightface, self._eva_clip, self._pulid, face_image_paths)

    def generate(
        self,
        prompt: str,
        *,
        face_image: str | None = None,
        face_images: list[str] | None = None,
        pulid_strength: float = 0.6,
        output_path: Path,
        **kwargs,
    ) -> Path:
        refs = face_images or ([face_image] if face_image else None)
        if not refs:
            raise ValueError(
                "pulid-flux2-klein requires 'face_image' or 'face_images' in the prompt frontmatter.\n"
                "Example: face_image: /path/to/reference.png\n"
                "         face_images: [/path/to/ref1.png, /path/to/ref2.png]"
            )

        self._load()
        id_tokens = self._extract_id_tokens(refs)

        # EVA-CLIP and InsightFace are done — move EVA-CLIP to CPU to free ~1 GB before inference
        if self._eva_clip is not None:
            self._eva_clip.to("cpu")
        torch.cuda.empty_cache()

        unpatch = patch_flux2(self._pipe.transformer, self._pulid, id_tokens, pulid_strength)
        try:
            return super().generate(prompt, output_path=output_path, **kwargs)
        finally:
            unpatch()
            # restore EVA-CLIP to GPU for next generation
            if self._eva_clip is not None:
                self._eva_clip.to("cuda")

    def cleanup(self) -> None:
        self._pulid = None
        self._insightface = None
        self._eva_clip = None
        super().cleanup()
