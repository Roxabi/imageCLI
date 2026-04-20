# PuLID Internals

## Why `compile=False` for all PuLID engines

`pulid-flux2-klein` (and `pulid-flux1-dev`) always runs with `compile=False` — `torch.compile` captures original `forward` methods at trace time and is incompatible with the per-generation transformer patching that PuLID uses.

## External Weights

- `~/ComfyUI/models/pulid/pulid_flux2_klein_v2.safetensors` (Klein)
- `~/ComfyUI/models/pulid/pulid_flux_v0.9.1.safetensors` (FLUX.1)
- InsightFace AntelopeV2 at `~/ComfyUI/models/insightface/`

Install: `uv sync --extra pulid`

## `pulid-flux2-klein` CA Loading

`from_safetensors()` remaps:
- `pulid_ca_double.N.*` → `double_ca.N.*`
- `pulid_ca_single.N.*` → `single_ca.N.*`

This fixes silent random-init that `strict=False` previously masked. CA module counts are auto-detected from weight keys (actual weights: 5 double + 7 single).

## `pulid-flux2-klein` Dim Projection

PuLID Klein v2 weights are dim=4096 (trained for Klein 9B); Klein 4B has hidden_size=3072.

`_make_projections()` creates orthogonal-init:
- `proj_up` (3072 → 4096)
- `proj_down` (4096 → 3072)

…so the trained CA attention patterns are preserved.

`_apply_ca()` runs:
```
hidden_states → proj_up → trained CA → proj_down
```

This differs from the iFayens ComfyUI approach, which discards all trained CA and uses random ones at 3072.

## `pulid-flux1-dev`

Uses GGUF Q5_K_S for the transformer (~6 GB) + PuLID v0.9.1 (20 CA modules, dim=3072). Monkey-patches `FluxTransformerBlock` / `FluxSingleTransformerBlock` `forward` methods. Always runs with `compile=False`.
