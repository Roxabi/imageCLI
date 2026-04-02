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
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)

_PULID_WEIGHTS = Path.home() / "ComfyUI/models/pulid/pulid_flux_v0.9.1.safetensors"
_INSIGHTFACE_DIR = Path.home() / "ComfyUI/models/insightface"

# Block schedule constants — match PuLID FLUX.1 reference implementation
_DOUBLE_INTERVAL = 2
_SINGLE_INTERVAL = 4
_N_DOUBLE = 19
_N_SINGLE = 38

# ceil(19/2)=10, ceil(38/4)=10 -> 20 CA modules total
_NUM_CA = math.ceil(_N_DOUBLE / _DOUBLE_INTERVAL) + math.ceil(_N_SINGLE / _SINGLE_INTERVAL)

# EVA-CLIP intermediate block indices (5 evenly-spaced layers across 24 ViT blocks)
_EVA_INTERMEDIATE_INDICES = [4, 8, 12, 16, 20]


# ── Inlined PuLID nn.Module classes ──────────────────────────────────────────
# These match the weights in pulid_flux_v0.9.1.safetensors exactly.
# Do NOT import from the PuLID repo — keep this file self-contained.


def _reshape_tensor(x: torch.Tensor, heads: int) -> torch.Tensor:
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    return x.reshape(bs, heads, length, -1)


class _PerceiverAttention(nn.Module):
    """Self+cross attention used inside IDFormer (query attends to context + self)."""

    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8, kv_dim: int | None = None):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        kv = kv_dim if kv_dim is not None else dim
        self.norm1 = nn.LayerNorm(kv)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, seq_len, _ = latents.shape
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = (_reshape_tensor(t, self.heads) for t in (q, k, v))
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
        return self.to_out(out)


def _FeedForward(dim: int, mult: int = 4) -> nn.Sequential:
    inner = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner, bias=False),
        nn.GELU(),
        nn.Linear(inner, dim, bias=False),
    )


class _IDFormer(nn.Module):
    """Perceiver-resampler identity encoder.

    Matches IDFormer(dim=1024, depth=10, dim_head=64, heads=16,
                     num_id_token=5, num_queries=32, output_dim=2048).

    Input:
        x  — id_cond  (B, 1280)  = concat[insightface_512, eva_cls_768]
        y  — id_vit_hidden  list[5] of (B, N, 1024)  multi-scale EVA features
    Output:
        (B, 32, 2048)  id_tokens consumed by PerceiverAttentionCA
    """

    def __init__(
        self,
        dim: int = 1024,
        depth: int = 10,
        dim_head: int = 64,
        heads: int = 16,
        num_id_token: int = 5,
        num_queries: int = 32,
        output_dim: int = 2048,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = dim**-0.5

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        _PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        _FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        for i in range(5):
            setattr(
                self,
                f"mapping_{i}",
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, dim),
                ),
            )

        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x: torch.Tensor, y: list[torch.Tensor]) -> torch.Tensor:
        latents = self.latents.repeat(x.size(0), 1, 1)
        num_duotu = x.shape[1] if x.ndim == 3 else 1
        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token * num_duotu, self.dim)
        latents = torch.cat((latents, x), dim=1)
        for i in range(5):
            vit_feature = getattr(self, f"mapping_{i}")(y[i])
            ctx_feature = torch.cat((x, vit_feature), dim=1)
            for attn, ff in self.layers[i * self.depth : (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents
        latents = latents[:, : self.num_queries]
        latents = latents @ self.proj_out
        return latents


class _PerceiverAttentionCA(nn.Module):
    """Cross-attention module that injects id_tokens into image hidden states.

    Matches PerceiverAttentionCA(dim=3072, dim_head=128, heads=16, kv_dim=2048).

    forward(x=id_tokens, latents=hidden_states) → correction (same shape as hidden_states)

    Note: the weight file uses the BFL argument order: forward(id, img) which maps to
    forward(x=id_tokens, latents=img_hidden_states). We preserve that convention.
    """

    def __init__(
        self, *, dim: int = 3072, dim_head: int = 128, heads: int = 16, kv_dim: int = 2048
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads  # 2048

        self.norm1 = nn.LayerNorm(kv_dim)  # norm for id_tokens (kv_dim=2048)
        self.norm2 = nn.LayerNorm(dim)  # norm for hidden_states (dim=3072)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # hidden -> queries
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)  # id -> k,v
        self.to_out = nn.Linear(inner_dim, dim, bias=False)  # -> dim=3072

    def forward(self, id_tokens: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            id_tokens:     (B, 32, 2048)  output of IDFormer
            hidden_states: (B, N, 3072)   image stream from transformer block
        Returns:
            correction:    (B, N, 3072)   to be added (scaled) to hidden_states
        """
        dtype = self.norm2.weight.dtype
        id_tokens = id_tokens.to(dtype)
        hidden_states = hidden_states.to(dtype)

        id_n = self.norm1(id_tokens)  # (B, 32, 2048)
        hs_n = self.norm2(hidden_states)  # (B, N, 3072)

        b, seq_len, _ = hidden_states.shape

        q = self.to_q(hs_n)  # (B, N, 2048)
        k, v = self.to_kv(id_n).chunk(2, dim=-1)  # each (B, 32, 2048)

        q, k, v = (_reshape_tensor(t, self.heads) for t in (q, k, v))
        # q: (B, heads, N, dim_head)  k,v: (B, heads, 32, dim_head)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # (B, heads, N, 32)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v  # (B, heads, N, dim_head)

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)  # (B, N, 2048)
        return self.to_out(out)  # (B, N, 3072)


class _PuLIDFlux1(nn.Module):
    """Container for PuLID FLUX.1 weights — IDFormer + 20 CA modules."""

    def __init__(self) -> None:
        super().__init__()
        self.pulid_encoder = _IDFormer()
        self.pulid_ca = nn.ModuleList([_PerceiverAttentionCA() for _ in range(_NUM_CA)])

    @classmethod
    def from_safetensors(cls, path: Path) -> "_PuLIDFlux1":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
        # Keys use 'pulid_encoder.*' and 'pulid_ca.*' prefixes — map directly
        model = cls()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("PuLID weights: missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("PuLID weights: unexpected keys: %s", unexpected[:5])
        return model


# ── transformer patching ──────────────────────────────────────────────────────


def _patch_flux1(
    transformer: object,
    pulid: _PuLIDFlux1,
    id_tokens: torch.Tensor,
    strength: float,
) -> object:
    """Monkey-patch FLUX.1-dev diffusers transformer blocks to inject PuLID identity.

    Double blocks (FluxTransformerBlock):
        - Every _DOUBLE_INTERVAL-th block (0, 2, 4, ...) gets a correction on
          hidden_states (image stream).
        - forward signature: (hidden_states, encoder_hidden_states, temb,
                               image_rotary_emb, joint_attention_kwargs)
          returns: (encoder_hidden_states, hidden_states)

    Single blocks (FluxSingleTransformerBlock):
        - Every _SINGLE_INTERVAL-th block (0, 4, 8, ...) gets a correction on
          hidden_states (image stream, AFTER txt split).
        - forward signature: same as double block
          returns: (encoder_hidden_states, hidden_states)
          Internally concatenates txt+img, processes them, then splits back.
          The returned hidden_states is already the image-only portion.

    Returns a callable that restores original forward methods.
    """
    dm = transformer
    double_blocks = list(dm.transformer_blocks)
    single_blocks = list(dm.single_transformer_blocks)

    orig_d: dict[int, object] = {}
    orig_s: dict[int, object] = {}

    ca_idx = 0

    for idx, block in enumerate(double_blocks):
        if idx % _DOUBLE_INTERVAL != 0:
            continue
        orig_d[idx] = block.forward
        local_ca_idx = ca_idx
        ca_idx += 1

        def make_double(i: int, ci: int):
            def patched(
                hidden_states: torch.Tensor = None,
                encoder_hidden_states: torch.Tensor = None,
                temb: torch.Tensor = None,
                image_rotary_emb=None,
                joint_attention_kwargs=None,
            ):
                enc_hs, img_hs = orig_d[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                correction = pulid.pulid_ca[ci](id_tokens, img_hs)
                return enc_hs, img_hs + strength * correction

            return patched

        block.forward = make_double(idx, local_ca_idx)

    for idx, block in enumerate(single_blocks):
        if idx % _SINGLE_INTERVAL != 0:
            continue
        orig_s[idx] = block.forward
        local_ca_idx = ca_idx
        ca_idx += 1

        def make_single(i: int, ci: int):
            def patched(
                hidden_states: torch.Tensor = None,
                encoder_hidden_states: torch.Tensor = None,
                temb: torch.Tensor = None,
                image_rotary_emb=None,
                joint_attention_kwargs=None,
            ):
                # FluxSingleTransformerBlock.forward internally concatenates
                # encoder_hidden_states + hidden_states, processes them, then
                # splits and returns (encoder_hidden_states, hidden_states).
                # The returned hidden_states is already the image-only stream.
                out_enc, out_img = orig_s[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                correction = pulid.pulid_ca[ci](id_tokens, out_img)
                return out_enc, out_img + strength * correction

            return patched

        block.forward = make_single(idx, local_ca_idx)

    def unpatch() -> None:
        for i, block in enumerate(double_blocks):
            if i in orig_d:
                block.forward = orig_d[i]
        for i, block in enumerate(single_blocks):
            if i in orig_s:
                block.forward = orig_s[i]

    return unpatch


# ── engine ─────────────────────────────────────────────────────────────────────

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
        self._pulid: _PuLIDFlux1 | None = None
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
        self._pulid = _PuLIDFlux1.from_safetensors(_PULID_WEIGHTS)
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
                self._eva_clip_trunk.to("cpu")
            if self._eva_clip_head is not None:
                self._eva_clip_head.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        unpatch = _patch_flux1(
            self._pipe.transformer, self._pulid, self._cached_id_tokens, pulid_strength
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
