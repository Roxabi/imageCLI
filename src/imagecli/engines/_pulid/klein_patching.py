"""Transformer-patching helpers for the FLUX.2-klein PuLID engine.

Dim mismatch handling (Strategy B — 2026-04-01):
    PuLID Klein v2 weights have dim=4096 (trained for Klein 9B).
    Klein 4B has hidden_size=3072. Rather than discard the trained CA weights
    (as iFayens does — creating random CA at 3072), we keep the trained CA at 4096
    and project hidden_states around them:
        hidden_states (3072) → proj_up (4096) → trained CA → proj_down (3072)
    The projection layers are random-init but the trained CA attention patterns
    that encode identity are preserved. This is theoretically stronger than
    the iFayens approach where only the IDFormer survives.

The projection pair is ephemeral — built at patch time inside ``patch_flux2``,
never attached to ``PuLIDFlux2`` as attributes. This keeps the trained-CA
weight load order (`PuLIDFlux2.from_safetensors` → `load_state_dict`) free
from any risk of a random-init regression on the projection layers.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .klein_modules import KleinPerceiverAttentionCA, PuLIDFlux2

logger = logging.getLogger(__name__)


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
    ca: KleinPerceiverAttentionCA,
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


def patch_flux2(
    transformer: object,
    pulid: PuLIDFlux2,
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
