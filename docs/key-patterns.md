# Key Patterns

- Registry ∈ `engine.py:_get_registry()` — add engines there
- `enable_model_cpu_offload()` on most (flux2-klein `generate` + 2-phase `batch`)
- Adaptive quantization ∈ `optimum-quanto`: fp8 (sm≥89 Ada/Blackwell) | int8 (Ampere sm≥80). SD3.5 T5 always int8.
- `_get_compute_capability()` ∈ `engine.py` detects GPU arch for quant selection
- `_optimize_pipe(pipe, compile=...)` ∈ `_load()` once; `_compiled` flag ¬double-compile
- Batch mode: 2-phase iff `supports_two_phase` (flux2-klein), else sequential
- `pulid-flux2-klein` always `compile=False` (captures forward methods, ¬compatible w/ per-gen patching)
- PuLID weights: `~/.roxabi/imagecli/weights/pulid/pulid_flux2_klein_v2.safetensors` + `pulid_flux_v0.9.1.safetensors` + InsightFace AntelopeV2 `~/.roxabi/imagecli/weights/insightface/`
