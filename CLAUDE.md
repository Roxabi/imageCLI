@.claude/stack.yml

# imageCLI

Unified CLI for local image generation. Supports FLUX.2-klein-4B, FLUX.1-dev, FLUX.1-schnell, and Stable Diffusion 3.5 Large Turbo backends via HuggingFace Diffusers.

## Tech Stack

- Python 3.12, managed with `uv`
- CLI framework: Typer + Rich
- Inference: HuggingFace `diffusers` + `optimum[quanto]` for fp8/int8 quantization
- GPU: PyTorch 2.7+ cu128 for RTX 5070 Ti (Blackwell sm_120, 16GB GDDR7); int8 fallback for Ampere (RTX 3080, sm_86)
- Linting: `ruff` (line-length 100, target py312)

## Engines

| Engine | Model | VRAM | Notes |
|---|---|---|---|
| `flux2-klein` | FLUX.2-klein-4B | ~8GB | Default. FP8 quanto + CPU offload. Peak VRAM: 7.84 GB. Black Forest Labs Nov 2025 |
| `pulid-flux2-klein` | FLUX.2-klein-4B + PuLID | ~9–10GB | Face identity lock. Requires `face_image` frontmatter + PuLID weights. `uv sync --extra pulid` |
| `pulid-flux1-dev` | FLUX.1-dev + PuLID | ~10GB | GGUF Q5_K_S + PuLID v0.9.1 face lock. 24 steps, 1024x1024. Requires `face_image` frontmatter |
| `flux1-dev` | FLUX.1-dev | ~10GB | fp8 (sm≥89) or int8 (Ampere) via optimum-quanto. Excellent quality |
| `flux1-schnell` | FLUX.1-schnell | ~10GB | fp8/int8 quantized. Apache 2.0, ungated. Fast 4-step generation |
| `sd35` | SD3.5 Large Turbo | ~14GB | Fast 20-step CFG-free. T5 encoder quantized to int8 |

## Project Layout

```
imagecli.example.toml     — copy to ~/imagecli.toml and customize
images/
  prompts_in/             — .md prompt files (tracked in git)
  images_out/             — generated images (gitignored)
src/imagecli/
  cli.py                  — Typer app: generate, batch, engines, info
  config.py               — TOML config loader (walks up from CWD to $HOME)
  engine.py               — Abstract ImageEngine base class + registry
  markdown.py             — YAML frontmatter parser for .md prompt files
  engines/
    flux2_klein.py        — FLUX.2-klein-4B engine (default)
    pulid_flux2_klein.py  — FLUX.2-klein-4B + PuLID face identity lock (optional extra)
    pulid_flux1_dev.py    — FLUX.1-dev GGUF + PuLID v0.9.1 face identity lock
    flux1_dev.py          — FLUX.1-dev quantized engine (fp8 on sm≥89, int8 on Ampere)
    flux1_schnell.py      — FLUX.1-schnell quantized engine (fp8 on sm≥89, int8 on Ampere)
    sd35.py               — SD3.5 Large Turbo engine
```

## All CLI Commands

```bash
# Single generation
imagecli generate "a cat in space"
imagecli generate "a cat in space" -e flux1-dev
imagecli generate "a cat in space" -e flux1-schnell
imagecli generate "a cat in space" -e sd35
imagecli generate prompt.md                      # from .md file with frontmatter
imagecli generate prompt.md -e flux1-dev         # override engine
imagecli generate "prompt" -W 1280 -H 720        # custom size
imagecli generate "prompt" --steps 30 --guidance 4.0
imagecli generate "prompt" --seed 42             # reproducible
imagecli generate "prompt" -n "blurry, ugly"     # negative prompt
imagecli generate "prompt" -o my_image.png       # custom output path
imagecli generate "prompt" --output-dir ./out    # custom output dir
imagecli generate "prompt" --no-compile          # skip torch.compile (faster startup)

# Batch generation (all .md files in a directory)
imagecli batch images/prompts_in/
imagecli batch images/prompts_in/ -e flux1-dev
imagecli batch images/prompts_in/ --output-dir ./results
imagecli batch images/prompts_in/ --no-compile

# Info
imagecli engines     # list engines with VRAM requirements
imagecli info        # show active config + GPU info
```

## Markdown Prompt Format

```markdown
---
engine: flux2-klein          # flux2-klein | pulid-flux2-klein | pulid-flux1-dev | flux1-dev | flux1-schnell | sd35
width: 1024                  # pixels, multiple of 64
height: 1024
steps: 50                    # inference steps (sd35: always 20, schnell: default 4)
guidance: 4.0                # CFG scale (sd35: always 1.0, schnell: always 0.0)
seed: 42                     # optional, for reproducibility
negative_prompt: "blurry"    # what to avoid
format: png                  # png | jpg | webp
face_image: /path/to/ref.png # pulid-flux2-klein only — reference face (abs or relative to .md file)
pulid_strength: 0.6          # pulid-flux2-klein only — identity lock strength (default 0.6)
---

Your prompt text here. Can be multiple paragraphs.
```

## User Config (`imagecli.toml`)

Searched by walking up from CWD to `$HOME`. Put at `~/imagecli.toml` for global defaults.

Priority: **CLI flag > .md frontmatter > imagecli.toml > hardcoded default**

## Memory Safety

`preflight_check(engine)` runs before every load and raises `MemoryError` (a `RuntimeError` subclass) if:

- Free VRAM < `engine.vram_gb` (checked via `torch.cuda.mem_get_info`)
- Free system RAM < `IMAGECLI_MIN_FREE_RAM_GB` env var (default `4.0`, reads `/proc/meminfo`)
- No CUDA GPU detected

`cleanup()` is called in a `finally` block after every generation: `torch.cuda.empty_cache()` + `gc.collect()`.

## Performance

`_optimize_pipe(pipe, compile=True)` is called once during `_load()` on every engine:

- `torch.set_float32_matmul_precision("high")` — TF32 tensor cores on Blackwell (~10-15% speedup)
- `pipe.vae.enable_tiling()` + `pipe.vae.enable_slicing()` — reduces peak VRAM for large images
- `torch.compile(transformer, mode="max-autotune-no-cudagraphs")` — kernel fusion (~20-40% speedup after first-run warmup); skipped with `--no-compile`
- `torch.cuda.set_per_process_memory_fraction(0.85)` — caps PyTorch VRAM at 85% of total
- All inference runs under `torch.inference_mode()`

`--no-compile` disables `torch.compile` on both `generate` and `batch`. Use it for faster startup or when testing.

## Key Patterns

- Engines are lazy-loaded — model loaded in `_load()` on first `generate()` call, not on import
- Engine registry in `engine.py:_get_registry()` — add new engines there
- `enable_model_cpu_offload()` used on all engines to handle VRAM pressure
- Adaptive quantization via `optimum-quanto`: fp8 on sm≥89 (Ada/Blackwell), int8 on Ampere (sm≥80). SD3.5 T5 always int8.
- `_get_compute_capability()` in `engine.py` detects GPU architecture for quantization selection
- `preflight_check()` runs before `_load()` — abort early rather than OOM mid-load
- `cleanup()` always runs in `finally` after generation (even on failure)
- `_optimize_pipe()` called once inside `_load()`; `_compiled` flag prevents double-compile
- Batch mode caches engine instances — one model load per engine across all files in the batch
- Output files never overwrite existing ones — suffix `_1`, `_2` etc. added automatically
- FLUX.2-klein is always the default (best quality-to-VRAM ratio for 16GB)
- `pulid-flux2-klein` always runs with `compile=False` — `torch.compile` captures original forward methods and is incompatible with per-generation transformer patching used by PuLID
- `pulid-flux2-klein` requires external model weights: `~/ComfyUI/models/pulid/pulid_flux2_klein_v2.safetensors` and InsightFace AntelopeV2 at `~/ComfyUI/models/insightface/`; install deps with `uv sync --extra pulid`
- `pulid-flux1-dev` uses GGUF Q5_K_S for the transformer (~6 GB) + PuLID v0.9.1 (20 CA modules, dim=3072). Monkey-patches FluxTransformerBlock/FluxSingleTransformerBlock forward methods. Always runs with `compile=False`.
- PuLID weights: `~/ComfyUI/models/pulid/pulid_flux_v0.9.1.safetensors` (FLUX.1) and `pulid_flux2_klein_v2.safetensors` (Klein, deprecated for avatars)

## Benchmark

Run these to record real VRAM and wall-clock numbers. Each generation prints `Peak VRAM (inference): X.XX GB` automatically.

```bash
# Smoke test (one image per engine)
imagecli generate "a white cat on a red chair" -e flux2-klein --seed 42
imagecli generate face_prompt.md               -e pulid-flux2-klein  # needs face_image frontmatter
imagecli generate face_prompt.md -e pulid-flux1-dev --no-compile  # needs face_image frontmatter
imagecli generate "a white cat on a red chair" -e flux1-dev   --seed 42
imagecli generate "a white cat on a red chair" -e flux1-schnell --seed 42
imagecli generate "a white cat on a red chair" -e sd35        --seed 42

# Full batch benchmark
imagecli batch images/prompts_in/ -e flux2-klein  --output-dir images/images_out/benchmark/klein
imagecli batch images/prompts_in/ -e flux1-dev    --output-dir images/images_out/benchmark/dev
imagecli batch images/prompts_in/ -e flux1-schnell --output-dir images/images_out/benchmark/schnell
imagecli batch images/prompts_in/ -e sd35         --output-dir images/images_out/benchmark/sd35
```

GPU support matrix:

| Engine | RTX 5070 Ti (16GB, sm_120) | RTX 3080 (10GB, sm_86) |
|---|---|---|
| `flux2-klein` | ✓ fp8 + cpu_offload | ✗ too large (~13GB) |
| `pulid-flux2-klein` | ✓ bf16 + cpu_offload | ✗ too large |
| `pulid-flux1-dev` | ✓ GGUF Q5_K_S + PuLID | ✗ too large |
| `flux1-dev` | ✓ fp8 | ✓ int8 |
| `flux1-schnell` | ✓ fp8 | ✓ int8 |
| `sd35` | ✓ int8 T5 | ✗ too large (~14GB) |

After running, update the Engines table above with real VRAM peaks from `imagecli info` + generation output.

## Conventions

- No over-engineering — thin CLI, flat and simple
- Heavy imports (torch, diffusers) deferred to engine `_load()` methods
- Output images go to `images/images_out/` by default
- Prompt files authored in `images/prompts_in/`
