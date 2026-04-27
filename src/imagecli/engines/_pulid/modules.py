"""Inlined PuLID nn.Module definitions.

These match the weights in ``pulid_flux_v0.9.1.safetensors`` exactly. The file
stays self-contained — no imports from the external PuLID repo — so the
safetensors state_dict loads without key rewriting.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Block schedule constants — match the PuLID FLUX.1 reference implementation.
# Also consumed by ``patching.patch_flux1``.
_DOUBLE_INTERVAL = 2
_SINGLE_INTERVAL = 4
_N_DOUBLE = 19
_N_SINGLE = 38

# ceil(19/2)=10, ceil(38/4)=10 -> 20 CA modules total.
_NUM_CA = math.ceil(_N_DOUBLE / _DOUBLE_INTERVAL) + math.ceil(_N_SINGLE / _SINGLE_INTERVAL)


def _reshape_tensor(x: torch.Tensor, heads: int) -> torch.Tensor:
    bs, length, _width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    return x.reshape(bs, heads, length, -1)


class PerceiverAttention(nn.Module):
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


class IDFormer(nn.Module):
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
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
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
            for attn, ff in cast(
                list[tuple[nn.Module, nn.Module]],
                list(self.layers[i * self.depth : (i + 1) * self.depth]),
            ):
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents
        latents = latents[:, : self.num_queries]
        latents = latents @ self.proj_out
        return latents


class PerceiverAttentionCA(nn.Module):
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


class PuLIDFlux1(nn.Module):
    """Container for PuLID FLUX.1 weights — IDFormer + 20 CA modules."""

    def __init__(self) -> None:
        super().__init__()
        self.pulid_encoder = IDFormer()
        self.pulid_ca = nn.ModuleList([PerceiverAttentionCA() for _ in range(_NUM_CA)])

    @classmethod
    def from_safetensors(cls, path: Path) -> "PuLIDFlux1":
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
