@.claude/stack.yml
@~/.claude/shared/global-patterns.md

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
images/prompts_in/        — .md prompts (git-tracked)
src/imagecli/             — cli, config, engine, markdown, daemon
src/imagecli/engines/     — flux2_klein, flux2_klein_fp8, flux2_klein_fp4,
                            pulid_flux2_klein, pulid_flux1_dev,
                            flux1_dev, flux1_schnell, sd35
```

## CLI (quick)

```
imagecli generate "text" | prompt.md  [-e ENGINE] [flags...]
imagecli batch DIR                    [-e ENGINE] [--two-phase] [flags...]
imagecli engines | info
```

→ `docs/cli.md` — flags + examples. `docs/prompt-format.md` — frontmatter. `docs/configuration.md` — `imagecli.toml`.

## Key Invariants

- Engines lazy: load ∈ `_load()` on 1st `generate()`, ¬on import
- `preflight_check()` pre-`_load()` — abort early, ¬OOM mid-load
- `cleanup()` ∈ `finally` post-gen (even on failure)
- Output: never overwrite existing — auto-suffix `_1`, `_2`, …
- Default: `flux2-klein` — best quality/VRAM @ 16GB

→ `docs/key-patterns.md` — full patterns. `docs/performance.md` — auto-mode, compile. `docs/memory-safety.md` — preflight details. `docs/benchmark.md` — measured results. `docs/lora.md` — training + inference. `docs/pulid-internals.md` — CA remapping, dim projection. `docs/QUADLET-DEPLOYMENT.md` — install runbook.

## Conventions

- ¬over-engineering — thin flat CLI
- Heavy imports (torch, diffusers) deferred to engine `_load()`
- Output → `~/.roxabi/imagecli/out/` (CLI); NATS satellite → HttpBlobStore (no local FS since #97) · prompts → `images/prompts_in/` (git-tracked)
