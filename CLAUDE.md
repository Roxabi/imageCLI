@.claude/stack.yml

# imageCLI

Unified CLI for local image generation. Supports FLUX.2-klein-4B, FLUX.1-dev, FLUX.1-schnell, and Stable Diffusion 3.5 Large Turbo backends via HuggingFace Diffusers.

## Tech Stack

- Python 3.12, managed with `uv`
- CLI framework: Typer + Rich
- Inference: HuggingFace `diffusers` + `optimum[quanto]` for fp8/int8 quantization
- GPU: PyTorch 2.7+ cu128 for RTX 5070 Ti (Blackwell sm_120, 16GB GDDR7)
- Linting: `ruff` (line-length 100, target py312)

## Engines

| Engine | Model | VRAM | Notes |
|---|---|---|---|
| `flux2-klein` | FLUX.2-klein-4B | ~13GB | Default. Best quality/VRAM ratio. Black Forest Labs Nov 2025 |
| `flux1-dev` | FLUX.1-dev | ~10GB | fp8 quantized via optimum-quanto. Excellent quality |
| `flux1-schnell` | FLUX.1-schnell | ~10GB | fp8 quantized. Apache 2.0, ungated. Fast 4-step generation |
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
    flux1_dev.py          — FLUX.1-dev fp8 quantized engine
    flux1_schnell.py      — FLUX.1-schnell fp8 quantized engine
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
engine: flux2-klein          # flux2-klein | flux1-dev | flux1-schnell | sd35
width: 1024                  # pixels, multiple of 64
height: 1024
steps: 50                    # inference steps (sd35: always 20, schnell: default 4)
guidance: 4.0                # CFG scale (sd35: always 1.0, schnell: always 0.0)
seed: 42                     # optional, for reproducibility
negative_prompt: "blurry"    # what to avoid
format: png                  # png | jpg | webp
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
- fp8 quantization via `optimum-quanto` (FLUX.1-dev/schnell transformer, SD3.5 T5 encoder)
- `preflight_check()` runs before `_load()` — abort early rather than OOM mid-load
- `cleanup()` always runs in `finally` after generation (even on failure)
- `_optimize_pipe()` called once inside `_load()`; `_compiled` flag prevents double-compile
- Batch mode caches engine instances — one model load per engine across all files in the batch
- Output files never overwrite existing ones — suffix `_1`, `_2` etc. added automatically
- FLUX.2-klein is always the default (best quality-to-VRAM ratio for 16GB)

## Conventions

- No over-engineering — thin CLI, flat and simple
- Heavy imports (torch, diffusers) deferred to engine `_load()` methods
- Output images go to `images/images_out/` by default
- Prompt files authored in `images/prompts_in/`
