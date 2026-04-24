"""PuLID nn.Modules for the FLUX.2-klein Klein-v2 engine.

Shapes match ``pulid_flux2_klein_v2.safetensors``: trained at dim=4096 for
Klein 9B. Weight remapping (``pulid_ca_double.N.*`` → ``double_ca.N.*``,
``pulid_ca_single.N.*`` → ``single_ca.N.*``) is handled in
``PuLIDFlux2.from_safetensors``. Dim mismatch against Klein 4B
(hidden_size=3072) is resolved ephemerally at patch time via the
projection pair in :mod:`klein_patching` — see that module's docstring
for the Strategy B rationale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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


class PuLIDFlux2(nn.Module):
    def __init__(self, dim: int = 4096, n_double_ca: int = 5, n_single_ca: int = 7):
        super().__init__()
        self.dim = dim
        self.id_former = _IDFormer(dim=dim)
        self.double_ca = nn.ModuleList([_PerceiverAttentionCA(dim=dim) for _ in range(n_double_ca)])
        self.single_ca = nn.ModuleList([_PerceiverAttentionCA(dim=dim) for _ in range(n_single_ca)])

    @classmethod
    def from_safetensors(cls, path: Path) -> "PuLIDFlux2":
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
