"""BFL → diffusers NVFP4 state-dict converter.

Converts BFL's native NVFP4 key layout (fused QKV, ``*_blocks.N.img_attn.qkv``)
to the diffusers layout (split Q/K/V, ``transformer_blocks.N.attn.to_q``) while
preserving NVFP4 tensor suffixes (``.weight``, ``.weight_scale``,
``.weight_scale_2``, ``.input_scale``).

Self-contained: torch is imported lazily inside the converter; no imagecli
imports.
"""

from __future__ import annotations

_RENAME = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
}

_DOUBLE_BLOCK_MAP = {
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

_SINGLE_BLOCK_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}

# NVFP4 tensor suffixes: each linear layer has these 4 tensors
NVFP4_SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")


def convert_nvfp4_state_dict(raw: dict[str, object]) -> dict[str, object]:
    """Convert BFL NVFP4 state dict keys to diffusers naming, splitting fused QKV."""
    import torch

    out: dict[str, object] = {}

    # Step 1: Simple renames
    for old_key in list(raw.keys()):
        new_key = old_key
        for old, new in _RENAME.items():
            new_key = new_key.replace(old, new)
        if new_key != old_key:
            raw[new_key] = raw.pop(old_key)

    # Step 2: Double blocks — handle QKV split and non-QKV remap
    for key in list(raw.keys()):
        if "double_blocks." not in key:
            continue

        parts = key.split(".")
        block_idx = parts[1]
        # Reconstruct the within-block name (everything between block idx and the last part)
        # e.g. "double_blocks.0.img_attn.qkv.weight" → within="img_attn.qkv", suffix=".weight"
        rest = ".".join(parts[2:])

        # Check for NVFP4 suffix
        suffix = None
        for s in NVFP4_SUFFIXES + (".scale",):
            if rest.endswith(s.lstrip(".")):
                suffix = rest[rest.rindex(s.lstrip(".")) - 1 :]  # include the dot
                within = rest[: rest.rindex(s.lstrip(".")) - 1]
                break
        if suffix is None:
            # Regular weight/bias
            within = ".".join(parts[2:-1])
            suffix = "." + parts[-1]

        if suffix == ".scale":
            suffix = ".weight"

        if "qkv" in within:
            # Fused QKV → split into Q, K, V (split dim=0 into 3 chunks)
            tensor = raw.pop(key)
            if tensor.dim() >= 1:
                chunks = torch.chunk(tensor, 3, dim=0)
            else:
                # Scalar (e.g. weight_scale_2, input_scale) — duplicate for each
                chunks = (tensor, tensor, tensor)

            if "img" in within:
                names = ("attn.to_q", "attn.to_k", "attn.to_v")
            else:
                names = ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")

            for name, chunk in zip(names, chunks):
                out_key = f"transformer_blocks.{block_idx}.{name}{suffix}"
                out[out_key] = chunk
        else:
            # Non-QKV — simple remap
            if within in _DOUBLE_BLOCK_MAP:
                new_within = _DOUBLE_BLOCK_MAP[within]
                out_key = f"transformer_blocks.{block_idx}.{new_within}{suffix}"
                out[out_key] = raw.pop(key)

    # Step 3: Single blocks — simple remap
    for key in list(raw.keys()):
        if "single_blocks." not in key:
            continue

        parts = key.split(".")
        block_idx = parts[1]
        rest = ".".join(parts[2:])

        suffix = None
        for s in NVFP4_SUFFIXES + (".scale",):
            if rest.endswith(s.lstrip(".")):
                suffix = rest[rest.rindex(s.lstrip(".")) - 1 :]
                within = rest[: rest.rindex(s.lstrip(".")) - 1]
                break
        if suffix is None:
            within = ".".join(parts[2:-1])
            suffix = "." + parts[-1]

        if suffix == ".scale":
            suffix = ".weight"

        if within in _SINGLE_BLOCK_MAP:
            new_within = _SINGLE_BLOCK_MAP[within]
            out_key = f"single_transformer_blocks.{block_idx}.{new_within}{suffix}"
            out[out_key] = raw.pop(key)

    # Step 4: Anything remaining (already renamed in step 1)
    for key, val in raw.items():
        if key not in out:
            out[key] = val

    return out
