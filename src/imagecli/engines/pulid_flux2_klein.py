"""FLUX.2-klein-4B + PuLID face identity lock engine.

Ported from iFayens/ComfyUI-PuLID-Flux2 — ComfyUI dependencies stripped,
runs directly in the imagecli diffusers pipeline.

Face reference image path must be supplied via ``face_image`` in frontmatter.

Dim mismatch handling (Strategy B — 2026-04-01):
    PuLID Klein v2 weights have dim=4096 (trained for Klein 9B).
    Klein 4B has hidden_size=3072. Rather than discard the trained CA weights
    (as iFayens does — creating random CA at 3072), we keep the trained CA at 4096
    and project hidden_states around them:
        hidden_states (3072) → proj_up (4096) → trained CA → proj_down (3072)
    The projection layers are random-init but the trained CA attention patterns
    that encode identity are preserved. This is theoretically stronger than
    the iFayens approach where only the IDFormer survives.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)

_PULID_DIR = Path.home() / "ComfyUI/models/pulid"
_INSIGHTFACE_DIR = Path.home() / "ComfyUI/models/insightface"
_PULID_DEFAULT = _PULID_DIR / "pulid_flux2_klein_v2.safetensors"


# ── PuLID nn.Module classes ────────────────────────────────────────────────────


class _PerceiverAttentionCA(nn.Module):
    def __init__(self, dim: int = 4096, dim_head: int = 64, heads: int = 16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        dtype = self.norm1.weight.dtype
        x, context = x.to(dtype), context.to(dtype)
        x_n, ctx = self.norm1(x), self.norm2(context)
        q = self.to_q(x_n)
        k, v = self.to_kv(ctx).chunk(2, dim=-1)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, self.heads, self.dim_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(reshape(q), reshape(k), reshape(v))
        return self.to_out(out.transpose(1, 2).contiguous().view(B, N, -1))


class _IDFormer(nn.Module):
    def __init__(self, dim: int = 4096, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(512 + 768, dim),
            nn.GELU(),
            nn.Linear(dim, dim * num_tokens),
        )
        self.latents = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.layers = nn.ModuleList([_PerceiverAttentionCA(dim=dim) for _ in range(4)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, id_embed: torch.Tensor, clip_embed: torch.Tensor) -> torch.Tensor:
        B = id_embed.shape[0]
        clip_embed = clip_embed - clip_embed.mean(dim=-1, keepdim=True)
        combined = torch.cat([id_embed, clip_embed], dim=-1)
        tokens = self.proj(combined).view(B, self.num_tokens, -1)
        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = latents + layer(latents, tokens)
        return self.norm(latents)


class _PuLIDFlux2(nn.Module):
    def __init__(self, dim: int = 4096, n_double_ca: int = 5, n_single_ca: int = 7):
        super().__init__()
        self.dim = dim
        self.id_former = _IDFormer(dim=dim)
        self.double_ca = nn.ModuleList([_PerceiverAttentionCA(dim=dim) for _ in range(n_double_ca)])
        self.single_ca = nn.ModuleList([_PerceiverAttentionCA(dim=dim) for _ in range(n_single_ca)])

    @classmethod
    def from_safetensors(cls, path: Path) -> "_PuLIDFlux2":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
        dim = state["id_former.latents"].shape[-1]

        # Auto-detect CA module counts from key structure.
        # Weights use pulid_ca_double.N.* / pulid_ca_single.N.* prefixes.
        n_double = (
            max(
                (int(k.split(".")[1]) for k in state if k.startswith("pulid_ca_double.")),
                default=-1,
            )
            + 1
        )
        n_single = (
            max(
                (int(k.split(".")[1]) for k in state if k.startswith("pulid_ca_single.")),
                default=-1,
            )
            + 1
        )
        logger.info(
            "PuLID weights: dim=%d, %d double CA, %d single CA",
            dim,
            n_double,
            n_single,
        )

        model = cls(dim=dim, n_double_ca=n_double, n_single_ca=n_single)

        # Remap key prefixes: pulid_ca_double → double_ca, pulid_ca_single → single_ca
        remapped: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            k_new = k.replace("pulid_ca_double.", "double_ca.").replace(
                "pulid_ca_single.", "single_ca."
            )
            remapped[k_new] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            logger.warning("PuLID load: missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("PuLID load: unexpected keys: %s", unexpected[:5])
        return model


# ── transformer patching ───────────────────────────────────────────────────────


def _get_flux_inner(model: object) -> object:
    """Unwrap ComfyUI model wrappers; diffusers transformers pass straight through."""
    if hasattr(model, "model"):
        model = model.model  # type: ignore[union-attr]
    if hasattr(model, "diffusion_model"):
        model = model.diffusion_model  # type: ignore[union-attr]
    return model


def _detect_variant(transformer: object) -> tuple[str, int, int, int]:
    """Return (variant_name, hidden_dim, n_double, n_single) for Klein 4B / 9B."""
    dm = _get_flux_inner(transformer)
    double_blocks = getattr(dm, "transformer_blocks", None) or getattr(dm, "double_blocks", [])
    single_blocks = getattr(dm, "single_transformer_blocks", None) or getattr(
        dm, "single_blocks", []
    )
    n_d, n_s = len(double_blocks), len(single_blocks)
    if n_d <= 6 and n_s <= 22:
        return "klein_4b", 3072, n_d, n_s
    return "klein_9b", 4096, n_d, n_s


def _ca_index(block_idx: int, total: int, num_ca: int) -> int:
    return block_idx if total <= num_ca else int(block_idx * num_ca / total)


def _scale(block_idx: int, total: int, block_type: str) -> float:
    p = block_idx / max(total, 1)
    if block_type == "double":
        return 8.0 if p < 0.4 else 5.0 if p < 0.7 else 3.0
    return 6.5 if p < 0.3 else 4.5 if p < 0.6 else 3.0 if p < 0.85 else 1.8


def _make_projections(
    pulid_dim: int,
    model_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Linear, nn.Linear] | None:
    """Create up/down projection pair if dims mismatch. Returns None if no projection needed."""
    if pulid_dim == model_dim:
        return None
    logger.info(
        "Dim mismatch: model=%d, PuLID=%d — creating projection layers", model_dim, pulid_dim
    )
    proj_up = nn.Linear(model_dim, pulid_dim, bias=False)
    proj_down = nn.Linear(pulid_dim, model_dim, bias=False)
    # Orthogonal init preserves norms better than random normal for pass-through projections
    nn.init.orthogonal_(proj_up.weight)
    nn.init.orthogonal_(proj_down.weight)
    proj_up.to(device, dtype=dtype)
    proj_down.to(device, dtype=dtype)
    return proj_up, proj_down


def _apply_ca(
    ca: _PerceiverAttentionCA,
    hidden_states: torch.Tensor,
    id_tokens: torch.Tensor,
    projections: tuple[nn.Linear, nn.Linear] | None,
) -> torch.Tensor:
    """Run CA with optional dim projection around trained weights."""
    if projections is not None:
        proj_up, proj_down = projections
        # hidden_states (3072) → proj_up (4096) → trained CA → proj_down (3072)
        correction = ca(proj_up(hidden_states), id_tokens)
        return F.normalize(proj_down(correction), p=2, dim=-1)
    return F.normalize(ca(hidden_states, id_tokens), p=2, dim=-1)


def _patch_flux(
    transformer: object,
    pulid: _PuLIDFlux2,
    id_tokens: torch.Tensor,
    strength: float,
) -> object:
    """Monkey-patch transformer blocks to inject PuLID identity. Returns unpatch callable."""
    dm = _get_flux_inner(transformer)
    double_blocks = getattr(dm, "transformer_blocks", None) or getattr(dm, "double_blocks", [])
    single_blocks = getattr(dm, "single_transformer_blocks", None) or getattr(
        dm, "single_blocks", []
    )
    n_d, n_s = len(double_blocks), len(single_blocks)

    # Detect dim mismatch and create projections if needed
    _, model_dim, _, _ = _detect_variant(transformer)
    projections = _make_projections(
        pulid.dim,
        model_dim,
        device=id_tokens.device,
        dtype=id_tokens.dtype,
    )

    orig_d: dict[int, object] = {}
    orig_s: dict[int, object] = {}

    for idx, block in enumerate(double_blocks):
        orig_d[idx] = block.forward

        def make_double(i: int):
            # Flux2TransformerBlock returns (encoder_hidden_states, hidden_states)
            # — text first, image second. PuLID correction targets the IMAGE stream.
            def patched(
                hidden_states=None,
                encoder_hidden_states=None,
                temb_mod_img=None,
                temb_mod_txt=None,
                **kwargs,
            ):
                enc_hs, img_hs = orig_d[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=temb_mod_img,
                    temb_mod_txt=temb_mod_txt,
                    **kwargs,
                )
                ca = pulid.double_ca[_ca_index(i, n_d, len(pulid.double_ca))]
                img_bf = img_hs.to(torch.bfloat16)
                correction = _apply_ca(ca, img_bf, id_tokens, projections)
                return enc_hs, img_bf + strength * _scale(i, n_d, "double") * correction

            return patched

        block.forward = make_double(idx)

    for idx, block in enumerate(single_blocks):
        orig_s[idx] = block.forward

        def make_single(i: int):
            # Flux2SingleTransformerBlock returns a SINGLE tensor (not a tuple)
            def patched(hidden_states=None, encoder_hidden_states=None, temb_mod=None, **kwargs):
                out_hs = orig_s[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod=temb_mod,
                    **kwargs,
                )
                ca = pulid.single_ca[_ca_index(i, n_s, len(pulid.single_ca))]
                out_bf = out_hs.to(torch.bfloat16)
                correction = _apply_ca(ca, out_bf, id_tokens, projections)
                return out_bf + strength * _scale(i, n_s, "single") * correction

            return patched

        block.forward = make_single(idx)

    def unpatch() -> None:
        for i, block in enumerate(double_blocks):
            if i in orig_d:
                block.forward = orig_d[i]
        for i, block in enumerate(single_blocks):
            if i in orig_s:
                block.forward = orig_s[i]

    return unpatch


# ── shared identity extraction ────────────────────────────────────────────────


def _extract_id_tokens(
    insightface: object,
    eva_clip: object,
    pulid: _PuLIDFlux2,
    face_image_path: str,
) -> torch.Tensor:
    """Extract PuLID identity tokens from a face reference image.

    Shared by PuLIDFlux2KleinEngine and PuLIDFlux2KleinFP4Engine so that any
    bug fix applies to both engines automatically.
    """
    from PIL import Image

    device = torch.device("cuda")
    dtype = torch.bfloat16

    img = np.array(Image.open(face_image_path).convert("RGB"))
    faces = insightface.get(img)  # type: ignore[union-attr]
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
        clip_out = eva_clip(face_t.float())  # type: ignore[union-attr]
        if isinstance(clip_out, (list, tuple)):
            clip_out = clip_out[0]
        if clip_out.dim() == 3:
            clip_out = clip_out[:, 0, :]
        clip_embed = clip_out.to(device, dtype=dtype)

        id_tokens = pulid.id_former(id_embed, clip_embed)
        id_tokens = F.normalize(id_tokens, p=2, dim=-1)

    return id_tokens


# ── engine ─────────────────────────────────────────────────────────────────────


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
        self._pulid: _PuLIDFlux2 | None = None
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
        logger.info("PuLID engine ready.")

    def _extract_id_tokens(self, face_image_path: str) -> torch.Tensor:
        return _extract_id_tokens(self._insightface, self._eva_clip, self._pulid, face_image_path)

    def generate(
        self,
        prompt: str,
        *,
        face_image: str | None = None,
        pulid_strength: float = 0.6,
        output_path: Path,
        **kwargs,
    ) -> Path:
        if not face_image:
            raise ValueError(
                "pulid-flux2-klein requires 'face_image' in the prompt frontmatter.\n"
                "Example: face_image: /path/to/reference.png"
            )

        self._load()
        id_tokens = self._extract_id_tokens(face_image)

        # EVA-CLIP and InsightFace are done — move EVA-CLIP to CPU to free ~1 GB before inference
        if self._eva_clip is not None:
            self._eva_clip.to("cpu")
        torch.cuda.empty_cache()

        unpatch = _patch_flux(self._pipe.transformer, self._pulid, id_tokens, pulid_strength)
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
