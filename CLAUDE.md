@.claude/stack.yml

# imageCLI

Local image gen CLI — FLUX.2-klein-4B, FLUX.1-dev/schnell, SD3.5 Large Turbo (HF Diffusers).

Python 3.12 via `uv` · Typer+Rich · PyTorch 2.11+ cu130 · ruff (L≤100, py312) · GPU: RTX 5070 Ti (Blackwell sm_120 16GB) | Ampere int8 fallback (RTX 3080 sm_86).

## Engines

| Engine | Model | VRAM | Notes |
|---|---|---|---|
| `flux2-klein` | FLUX.2-klein-4B | ~8GB | **Default, fastest.** FP8 quanto + Marlin GEMM (~7 it/s). Single: CPU offload (~8GB). Batch: all-on-GPU (~12GB) ∨ `--two-phase` (~8GB). |
| `flux2-klein-fp8` | FLUX.2-klein-4B FP8 | ~13GB | torchao FP8 weight-only (~4.3 it/s). ~40% slower vs quanto (dequant overhead). `torch.compile` compatible. `uv sync --group fp8` |
| `flux2-klein-fp4` | FLUX.2-klein-4B NVFP4 | ~11.5GB | Blackwell FP4 via comfy-kitchen (~6.8 it/s). ~2GB transformer. Req sm_120+ cu130. `uv sync --group fp4` |
| `pulid-flux2-klein` | FLUX.2-klein-4B + PuLID | ~9-10GB | Face identity lock. Req `face_image` frontmatter + PuLID weights. `uv sync --extra pulid` |
| `pulid-flux1-dev` | FLUX.1-dev + PuLID | ~10GB | GGUF Q5_K_S + PuLID v0.9.1. 24 steps, 1024². Req `face_image` frontmatter |
| `flux1-dev` | FLUX.1-dev | ~10GB | fp8 (sm≥89) ∨ int8 (Ampere) via optimum-quanto. Excellent quality |
| `flux1-schnell` | FLUX.1-schnell | ~10GB | fp8/int8. Apache 2.0, ungated. Fast 4-step |
| `sd35` | SD3.5 Large Turbo | ~14GB | 20-step CFG-free. T5 int8 |

## Project Layout

```
imagecli.example.toml     — copy → ~/imagecli.toml
images/
  prompts_in/             — .md prompts (git-tracked)
  images_out/             — generated (gitignored)
src/imagecli/
  cli.py                  — Typer app: generate, batch, engines, info
  config.py               — TOML loader (walks CWD → $HOME)
  engine.py               — `ImageEngine` ABC + registry
  markdown.py             — YAML frontmatter parser
  daemon.py               — NATS adapter / socket daemon (`imagecli serve`)
  engines/
    flux2_klein.py        — default, quanto FP8
    flux2_klein_fp8.py    — torchao FP8 (40% slower, torch.compile OK)
    flux2_klein_fp4.py    — NVFP4 (Blackwell, comfy-kitchen)
    pulid_flux2_klein.py  — Klein + PuLID face lock (extra)
    pulid_flux1_dev.py    — FLUX.1-dev GGUF + PuLID v0.9.1
    flux1_dev.py          — FLUX.1-dev quanto (fp8 sm≥89 | int8 Ampere)
    flux1_schnell.py      — FLUX.1-schnell quanto
    sd35.py               — SD3.5 Large Turbo
```

## CLI (quick)

```
imagecli generate "text" | prompt.md  [-e ENGINE] [flags...]
imagecli batch DIR                    [-e ENGINE] [--two-phase] [flags...]
imagecli engines | info
```

→ `docs/cli.md` — full flag reference + examples. Or `imagecli --help`.

## Markdown Prompt Format

```markdown
---
engine: flux2-klein          # ∈ ENGINE set above
width: 1024                  # multiple of 64
height: 1024
steps: 50                    # sd35:20, schnell:4
guidance: 4.0                # sd35:1.0, schnell:0.0
seed: 42                     # optional
negative_prompt: "blurry"
format: png                  # png | jpg | webp
face_image: /path/to/ref.png # pulid-* only (abs or relative to .md)
pulid_strength: 0.6          # pulid-flux2-klein only (default 0.6)
lora_path: /path/to/lora.safetensors  # flux2-klein, flux2-klein-fp4, flux2-klein-fp8
lora_scale: 1.0              # default 1.0, try 1.5 for stronger identity
trigger: lyraface            # pivotal-tuning trigger (required if LoRA has emb_params)
embedding_path: /path/to/emb.safetensors  # standalone pivotal emb (overrides emb_params)
---

Prompt text. Can be multi-paragraph.
```

## Config

`imagecli.toml` searched CWD → `$HOME`. Global: `~/imagecli.toml`.
Priority: CLI flag > frontmatter > imagecli.toml > default.

## flux2-klein auto-mode

| Command | Mode | Peak VRAM | Compile |
|---|---|---|---|
| `generate` | CPU offload | ~8 GB | No |
| `batch` (default) | all-on-GPU | ~12 GB | No (quanto) / Yes (fp8/fp4) |
| `batch --two-phase` | 2-phase | ~8 GB enc → ~4 GB gen | No (quanto) / Yes (fp8/fp4) |

→ `docs/performance.md` — `_optimize_pipe()`, `torch.compile` status, 2-phase details, `--no-compile`.

## Memory Safety

`preflight_check(engine)` ∈ every `_load()` → raises `MemoryError` iff free VRAM < `engine.vram_gb` ∨ free RAM < `IMAGECLI_MIN_FREE_RAM_GB` (default 4.0) ∨ no CUDA. `cleanup()` ∈ `finally` post-gen.

→ `docs/memory-safety.md` — full details.

## Key Patterns

- Engines lazy: load ∈ `_load()` on 1st `generate()`, ¬on import
- Registry ∈ `engine.py:_get_registry()` — add engines there
- `enable_model_cpu_offload()` on most (flux2-klein `generate` + 2-phase `batch`)
- Adaptive quantization ∈ `optimum-quanto`: fp8 (sm≥89 Ada/Blackwell) | int8 (Ampere sm≥80). SD3.5 T5 always int8.
- `_get_compute_capability()` ∈ `engine.py` detects GPU arch for quant selection
- `preflight_check()` pre-`_load()` — abort early, ¬OOM mid-load
- `cleanup()` ∈ `finally` post-gen (even on failure)
- `_optimize_pipe(pipe, compile=...)` ∈ `_load()` once; `_compiled` flag ¬double-compile
- Batch mode: 2-phase iff `supports_two_phase` (flux2-klein), else sequential
- Output: never overwrite existing — auto-suffix `_1`, `_2`, …
- Default: `flux2-klein` — best quality/VRAM @ 16GB
- `pulid-flux2-klein` always `compile=False` (captures forward methods, ¬compatible w/ per-gen patching)
- PuLID weights: `~/ComfyUI/models/pulid/pulid_flux2_klein_v2.safetensors` + `pulid_flux_v0.9.1.safetensors` + InsightFace AntelopeV2 `~/ComfyUI/models/insightface/`

→ `docs/pulid-internals.md` — CA remapping, dim projection (3072↔4096), GGUF details.

## Benchmark

Measured: RTX 5070 Ti, Apr 2026. Summary: `quanto FP8` fastest @ 512² (7.05 it/s); `NVFP4 --two-phase` wins everything @ 1024²+ (only engine that fits 2048²).

→ `docs/benchmark.md` — smoke/batch commands, GPU support matrix, measured results, resolution scaling, engine-pick decision table, key findings.

## LoRA

Training: [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit/) externally.
Inference: `flux2-klein` ∨ `flux2-klein-fp8` via `--lora` | `lora_path`. Fused into base pre-FP8 quant.
Supported: quanto FP8 + torchao FP8. FP4 (pre-quantized) ¬supported.

→ `docs/lora.md` — config, load order, tuning.

## Conventions

- ¬over-engineering — thin flat CLI
- Heavy imports (torch, diffusers) deferred to engine `_load()`
- Output → `images/images_out/` default · prompts → `images/prompts_in/`
