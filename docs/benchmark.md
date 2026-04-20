# Benchmark

Run these to record real VRAM and wall-clock numbers. Each generation prints `Peak VRAM (inference): X.XX GB` automatically.

## Smoke Test

```bash
for E in flux2-klein flux2-klein-fp8 flux2-klein-fp4 flux1-dev flux1-schnell sd35; do
  imagecli generate "a white cat on a red chair" -e "$E" --seed 42
done
imagecli generate face_prompt.md -e pulid-flux2-klein            # needs face_image
imagecli generate face_prompt.md -e pulid-flux1-dev --no-compile # needs face_image
```

## Full Batch

```bash
for E in flux2-klein flux1-dev flux1-schnell sd35; do
  imagecli batch images/prompts_in/ -e "$E" --output-dir images/images_out/benchmark/$E
done
```

## GPU Support Matrix

| Engine | RTX 5070 Ti (16GB, sm_120) | RTX 3080 (10GB, sm_86) |
|---|---|---|
| `flux2-klein` | ✓ fp8 quanto + offload/all-on-GPU/2-phase | ✗ too large (~13GB) |
| `flux2-klein-fp8` | ✓ torchao FP8 (slower than quanto) | ✗ too large |
| `flux2-klein-fp4` | ✓ NVFP4 via comfy-kitchen (sm_120+ cu130) | ✗ requires sm_120+ |
| `pulid-flux2-klein` | ✓ bf16 + cpu_offload | ✗ too large |
| `pulid-flux1-dev` | ✓ GGUF Q5_K_S + PuLID | ✗ too large |
| `flux1-dev` | ✓ fp8 | ✓ int8 |
| `flux1-schnell` | ✓ fp8 | ✓ int8 |
| `sd35` | ✓ int8 T5 | ✗ too large (~14GB) |

## Measured Results (RTX 5070 Ti, 512x512, 28 steps, April 2026)

### Single image (cold start — includes model load)

| Engine | Total wall-clock | Steady it/s | Peak VRAM |
|---|---|---|---|
| quanto FP8 (CPU offload) | 28.5s | 7.05 | 7.84 GB |
| **NVFP4 (all-on-GPU)** | **15.2s** | 6.8 | 11.46 GB |

### Batch (model already loaded)

| Engine | Mode | Steady it/s | Gen VRAM | Free VRAM |
|---|---|---|---|---|
| quanto FP8 | all-on-GPU | **7.0** | 12.26 GB | 3.2 GB |
| quanto FP8 | --two-phase | 6.95 | 4.93 GB | 10.5 GB |
| NVFP4 | all-on-GPU | 6.78 | 11.14 GB | 4.3 GB |
| NVFP4 | --two-phase | 6.72 | **3.56 GB** | **11.9 GB** |
| torchao FP8 | all-on-GPU | 4.34 | 12.75 GB | 2.7 GB |

Multi-image per step (batch_size 2-4): zero speedup at 512x512 — GPU already compute-saturated.

### Resolution scaling

| Resolution | quanto FP8 best | NVFP4 best | Notes |
|---|---|---|---|
| 512x512 | **7.05 it/s** (all-on-GPU) | 6.78 it/s | quanto fastest at low res |
| 1024x1024 | 2.02 it/s (two-phase, 8.30 GB) | **2.13 it/s** (two-phase, 7.44 GB) | NVFP4 faster + less VRAM |
| 2048x2048 | OOM (all modes) | **0.41 it/s** (two-phase, 8.82 GB) | **NVFP4 only engine that fits** |

### Which engine to pick

| Scenario | Best pick | Why |
|---|---|---|
| 1 image (any res) | **NVFP4** | Half the cold start (15s vs 28s) — no runtime quantization |
| Batch 512x512 | **quanto FP8 all-on-GPU** | Fastest steady-state (7.0 it/s) |
| Batch 1024x1024 | **NVFP4 --two-phase** | Faster (2.13 vs 2.02 it/s) and less VRAM (7.44 vs 8.30 GB) |
| Batch 2048x2048 | **NVFP4 --two-phase** | Only option that fits in 16 GB (8.82 GB peak) |
| 500+ sharing GPU | **NVFP4 --two-phase** | Only 3.56 GB during gen at 512x512, 12 GB free for Qwen/other |

### Key findings

- **quanto FP8 + Marlin GEMM is fastest at 512x512** — native FP8 matmul, no dequant overhead
- **NVFP4 wins at 1024x1024+** — smaller transformer (~2 GB vs ~4 GB) leaves more VRAM for latents
- **NVFP4 is the only engine for 2048x2048** — quanto OOMs in all modes
- **NVFP4 wins cold start** — pre-quantized weights load in ~7s vs ~20s for quanto runtime quantization
- **NVFP4 --two-phase is the VRAM champion** — 3.56 GB during generation at 512x512, 7.44 GB at 1024x1024
- **torchao FP8 is 40% slower** — weight-only FP8 dequantizes to bf16 per matmul
- **--two-phase is free** — same speed as all-on-GPU, just lower VRAM during generation
- **torch.compile is a dead end** — quanto's QLinear patch blocks it; torchao unblocks it but is already slower
