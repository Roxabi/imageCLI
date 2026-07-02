# Performance

## Common Pipeline Optimization

`_optimize_pipe(pipe, compile=...)` is called once during `_load()` on every engine:

- `torch.set_float32_matmul_precision("high")` — TF32 tensor cores on Blackwell (~10-15% speedup)
- `pipe.vae.enable_tiling()` + `pipe.vae.enable_slicing()` — reduces peak VRAM for large images
- `torch.compile(transformer, mode="max-autotune-no-cudagraphs")` — kernel fusion (~20-40% speedup after first-run warmup); skipped with `--no-compile` or `compile=False`
- All inference runs under `torch.inference_mode()`

`--no-compile` disables `torch.compile` on both `generate` and `batch`. Use it for faster startup or when testing.

## flux2-klein: Automatic Mode Selection

`flux2-klein` uses different VRAM strategies depending on the command:

| Command | Mode | Peak VRAM | Compile | Why |
|---|---|---|---|---|
| `generate` | CPU offload | ~8 GB | No | Faster for 1-2 images — no compile warmup, no GPU load overhead |
| `batch` (default) | All-on-GPU | ~12 GB | No (quanto) / Yes (fp8/fp4) | Encoder + transformer + VAE all on GPU. No phase switching. Fastest |
| `batch --two-phase` | 2-phase | ~8 GB (encode) → ~4 GB (generate) | No (quanto) / Yes (fp8/fp4) | Lower VRAM. Use when running alongside other GPU processes |

### All-on-GPU batch (default)

Encoder (~8 GB) + transformer FP8 (~3.9 GB) + VAE (~0.17 GB) all on GPU at once (~12 GB). Each prompt is encoded and generated immediately — no model shuffling.

### 2-phase batch (`--two-phase`)

1. **Phase 1 (encode):** Text encoder loaded to GPU (~8 GB). All prompts encoded, embeddings cached on CPU. Encoder offloaded.
2. **Phase 2 (generate):** Transformer FP8 (~3.9 GB) + VAE (~0.17 GB) loaded to GPU. All images generated from cached embeddings.

### torch.compile status

Works with `flux2-klein-fp8` (no QLinear patch). Blocked on `flux2-klein` (quanto QLinear contiguity patch conflicts with compile). However, quanto's Marlin FP8 GEMM kernels are already faster than torchao+compile, so compile is not a net win currently.

### Two-phase support

Engines that set `supports_two_phase = True` get this optimization. Other engines (or mixed-engine batches) fall back to sequential per-image generation.
