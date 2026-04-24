"""FLUX.1-dev transformer monkey-patching for PuLID identity injection.

``patch_flux1`` rewrites the ``forward`` methods on selected double and single
transformer blocks to add a PuLID CA correction to the image stream. The
patching is per-generation and reversible — the returned ``unpatch`` callable
restores the original forwards. Incompatible with ``torch.compile`` since
``compile`` captures the original forward method bodies.
"""

from __future__ import annotations

import torch

from .modules import _DOUBLE_INTERVAL, _SINGLE_INTERVAL, PuLIDFlux1


def patch_flux1(
    transformer: object,
    pulid: PuLIDFlux1,
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
