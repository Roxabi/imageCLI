# imageCLI

![Python](https://img.shields.io/badge/python-3.11--3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)
![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white)
![version](https://img.shields.io/badge/version-0.1.0-22c55e)

Unified CLI for local image generation — FLUX.2-klein, FLUX.1-dev (+ PuLID face lock), FLUX.1-schnell, and SD3.5 Large Turbo backends via HuggingFace Diffusers.

## Why

Running image generation locally means no API costs, no rate limits, and full control over prompts and outputs. But wiring up HuggingFace Diffusers — quantization, VRAM management, preflight checks — is repetitive boilerplate.

imageCLI wraps all of that into a single command. Point it at a text prompt or a `.md` file and get an image. Swap backends with `-e`. Batch an entire directory. Import it as a Python library.

## Requirements

- Python 3.11–3.12
- CUDA GPU (16GB VRAM recommended; 10GB minimum for fp8 engines)
- CUDA 13.0 (cu130) for NVFP4 engine; CUDA 12.8+ for all others
- [uv](https://docs.astral.sh/uv/) package manager

## Install

### From source (development)

```bash
git clone <repo-url> && cd imageCLI
uv sync
cp imagecli.example.toml ~/imagecli.toml   # optional — customize defaults
```

Run via `uv run imagecli <command>` from the project directory.

### Global install (run from any directory)

imagecli ships a `[project.scripts]` entry so it installs as a proper system command.

**uv tool (preferred)** — run from the project directory so uv picks up the custom PyTorch `cu128` index from `pyproject.toml` automatically:

```bash
cd /path/to/imageCLI
uv tool install .
```

**pipx (alternative)** — pass the PyTorch index explicitly using `--index-url` (replaces PyPI entirely, avoiding dependency confusion for `torch`/`torchvision`):

```bash
pipx install -e /path/to/imageCLI --pip-args="--index-url https://download.pytorch.org/whl/cu128"
```

> **Note:** pip does not read `[tool.uv.sources]` from `pyproject.toml`. This means the git-pinned `diffusers` source and index overrides used by `uv` are ignored — pip resolves all packages from the pytorch wheel index above. For the exact tested dependency graph, prefer `uv tool install .`.

After either install, `imagecli` is on your `PATH` and works from any directory:

```bash
imagecli info                          # verify install + GPU detection
imagecli generate "a cat in space"     # generate from anywhere
```

Models are downloaded automatically on first use from HuggingFace (~13GB for FLUX.2-klein).

## Quick Start

```bash
# Generate with default engine (FLUX.2-klein)
imagecli generate "a cat in space"

# Use a specific engine
imagecli generate "a cat in space" -e flux1-dev

# Generate from a .md prompt file
imagecli generate images/prompts_in/lyra-v1-cinematic.md

# Batch all prompts in a directory
imagecli batch images/prompts_in/

# List available engines
imagecli engines

# Show active config and GPU info
imagecli info
```

## How it works

A prompt (text or `.md` file) is dispatched to an engine. The engine lazy-loads its model on the first call, runs preflight VRAM/RAM checks, generates the image, and saves it — then frees GPU memory.

```mermaid
flowchart LR
  A["prompt / .md file"] --> B["engine registry"]
  B --> C["preflight_check\nVRAM + RAM"]
  C --> D["_load()\nlazy model load"]
  D --> E["inference\ntorch.inference_mode()"]
  E --> F["image.save()"]
  F --> G["cleanup()\nVRAM freed"]
```

Engines stay alive across batch runs — the model loads once per session. `torch.compile` fuses CUDA kernels on the first generation, then runs at full speed for all subsequent images.

## Library API

imageCLI is also a Python library. Install it as a dependency and call it directly — no subprocess, no HTTP server.

```bash
# From another project
uv add --editable /path/to/imageCLI
```

```python
from pathlib import Path
from imagecli import generate, list_engines, preflight_check

# Generate an image — returns the path to the saved file
out: Path = generate(
    "a serene mountain landscape at sunset",
    engine="flux2-klein",       # optional, defaults to imagecli.toml
    width=1280,
    height=720,
    seed=42,                    # reproducible output
    output_dir="images/out/",
)
print(out)   # Path("images/out/image_20260312_143201.png")

# List available engines
for engine in list_engines():
    print(engine)   # "flux2-klein", "flux1-dev", …

# Check GPU / resource availability before loading a model
from imagecli import get_engine
eng = get_engine("flux2-klein")
preflight_check(eng)   # raises InsufficientResourcesError if VRAM too low
```

Public API (`__all__`): `generate`, `get_engine`, `list_engines`, `preflight_check`, `load_config`, `parse_prompt_file`, `ImageEngine`, `InsufficientResourcesError`, `PromptDoc`.

## Engines

| Engine | Model | VRAM | Steps | Speed | Notes |
|---|---|---|---|---|---|
| `flux2-klein` | FLUX.2-klein-4B | ~8GB | 50 | ~7s/step | **Default.** FP8 quanto + Marlin GEMM. Single: CPU offload. Batch: all-on-GPU (~12 GB) or `--two-phase` (~8 GB). Fastest. |
| `flux2-klein-fp8` | FLUX.2-klein-4B FP8 | ~13GB | 50 | ~4s/step | torchao FP8 weight-only. No quanto, torch.compile compatible. **~40% slower** than quanto (dequant overhead). `uv sync --group fp8` |
| `flux2-klein-fp4` | FLUX.2-klein-4B NVFP4 | ~11.5GB | 50 | ~6.8s/step | Blackwell FP4 via comfy-kitchen. ~2 GB transformer. Requires sm_120+ and cu130. `uv sync --group fp4` |
| `pulid-flux2-klein` | FLUX.2-klein-4B + PuLID | ~9–10GB | 50 | ~25s | Face identity lock (Klein). Requires `face_image` in frontmatter. `uv sync --extra pulid` |
| `pulid-flux1-dev` | FLUX.1-dev + PuLID v0.9.1 | ~10GB | 24 | ~42s | Face identity lock (recommended). GGUF Q5_K_S + PuLID. Clean 1024×1024, no banding. `uv sync --extra pulid` |
| `flux1-dev` | FLUX.1-dev fp8 | ~10GB | 50 | ~30s | Maximum quality, longer prompts. fp8 via optimum-quanto |
| `flux1-schnell` | FLUX.1-schnell fp8 | ~10GB | 4 | ~5s | Fastest. Apache 2.0, ungated. fp8 via optimum-quanto |
| `sd35` | SD3.5 Large Turbo | ~14GB | 20 | ~8s | CFG-free, realistic photos. T5 encoder quantized to int8 |

### Benchmarks (RTX 5070 Ti, 512x512, 28 steps, April 2026)

**Single image (cold start — includes model load):**

| Engine | Total wall-clock | Steady it/s | Peak VRAM |
|--------|-----------------|------------|-----------|
| `flux2-klein` (quanto FP8) | 28.5s | 7.05 | 7.84 GB |
| `flux2-klein-fp4` (NVFP4) | **15.2s** | 6.8 | 11.46 GB |

**Batch (model already loaded):**

| Engine | Mode | Steady it/s | Gen VRAM | Free VRAM |
|--------|------|------------|----------|-----------|
| `flux2-klein` | all-on-GPU | **7.0** | 12.26 GB | 3.2 GB |
| `flux2-klein` | --two-phase | 6.95 | 4.93 GB | 10.5 GB |
| `flux2-klein-fp4` | all-on-GPU | 6.78 | 11.14 GB | 4.3 GB |
| `flux2-klein-fp4` | --two-phase | 6.72 | **3.56 GB** | **11.9 GB** |

Multi-image per step (batch_size 2-4) provides zero speedup — GPU is already compute-saturated at 512x512.

**Resolution scaling:**

| Resolution | quanto FP8 best | NVFP4 best | Notes |
|-----------|----------------|------------|-------|
| 512x512 | **7.05 it/s** (all-on-GPU) | 6.78 it/s | quanto fastest at low res |
| 1024x1024 | 2.02 it/s (two-phase) | **2.13 it/s** (two-phase) | NVFP4 pulls ahead — smaller transformer leaves VRAM for latents |
| 2048x2048 | OOM (all modes) | **0.41 it/s** (two-phase only) | **NVFP4 is the only engine that fits** — 8.82 GB peak |

**Which engine to pick:**

| Scenario | Best pick | Why |
|----------|----------|-----|
| 1 image (any res) | **NVFP4** | Half the cold start (15s vs 28s) — no runtime quantization |
| Batch 512x512 | **quanto FP8 all-on-GPU** | Fastest steady-state (7.0 it/s) |
| Batch 1024x1024 | **NVFP4 --two-phase** | Faster (2.13 vs 2.02 it/s) and less VRAM (7.44 vs 8.30 GB) |
| Batch 2048x2048 | **NVFP4 --two-phase** | Only option that fits in 16 GB (8.82 GB peak) |
| 500+ sharing GPU | **NVFP4 --two-phase** | Only 3.56 GB during gen at 512x512, 12 GB free for other processes |

## Commands

### `generate` — Single image

```bash
imagecli generate "a cat in space"
imagecli generate "a cat in space" -e flux1-schnell
imagecli generate prompt.md                          # from .md file with frontmatter
imagecli generate prompt.md -e flux1-dev             # override engine from frontmatter
imagecli generate "prompt" -W 1280 -H 720            # custom dimensions
imagecli generate "prompt" --steps 30 --guidance 4.0
imagecli generate "prompt" --seed 42                 # reproducible output
imagecli generate "prompt" -n "blurry, ugly"         # negative prompt
imagecli generate "prompt" -o my_image.png           # explicit output path
imagecli generate "prompt" --output-dir ./out        # custom output directory
imagecli generate "prompt" --no-compile              # skip torch.compile for one-offs
```

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--engine` | `-e` | Engine name (`flux2-klein`, `flux2-klein-fp8`, `flux2-klein-fp4`, `pulid-flux2-klein`, `pulid-flux1-dev`, `flux1-dev`, `flux1-schnell`, `sd35`) | `flux2-klein` |
| `--width` | `-W` | Output width in pixels (multiple of 64) | `1024` |
| `--height` | `-H` | Output height in pixels (multiple of 64) | `1024` |
| `--steps` | `-s` | Inference steps | `50` |
| `--guidance` | `-g` | Guidance scale (CFG) | `4.0` |
| `--seed` | | Fixed seed for reproducibility | random |
| `--negative` | `-n` | Negative prompt | `""` |
| `--output` | `-o` | Output file path | auto-generated |
| `--output-dir` | | Output directory | `images/images_out` |
| `--format` | | Output format: `png`, `jpg`, `webp` | `png` |
| `--no-compile` | | Skip torch.compile (faster startup, slower generation) | off |

### `batch` — All prompts in a directory

```bash
imagecli batch images/prompts_in/
imagecli batch images/prompts_in/ -e flux1-dev
imagecli batch images/prompts_in/ --output-dir ./results
imagecli batch images/prompts_in/ --no-compile
```

Processes every `.md` file in the directory in sorted order.

**Batch modes (flux2-klein engines):** When all files use the same engine, batch mode picks the fastest strategy:

- **All-on-GPU (default):** Encoder + transformer + VAE all on GPU at once (~12 GB). No model shuffling. Fastest.
- **Two-phase (`--two-phase`):** Phase 1 loads text encoder (~8 GB), encodes all prompts, offloads. Phase 2 loads transformer + VAE (~4 GB), generates all images. Lower VRAM, useful when running alongside other GPU processes.

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--engine` | `-e` | Override engine for all files | per-file frontmatter |
| `--output-dir` | | Output directory for all images | `images/images_out` |
| `--no-compile` | | Skip torch.compile | off |
| `--two-phase` | | Force 2-phase batch (lower VRAM, slower) | off |

### `engines` — List engines

```bash
imagecli engines
```

Prints a table of all registered engines with VRAM requirements, model ID, and description.

### `info` — Config and GPU

```bash
imagecli info
```

Shows the active `imagecli.toml` values and detected GPU name + total VRAM.

## Markdown Prompt Format

Write prompts as `.md` files with YAML frontmatter. Store them in `images/prompts_in/` (tracked in git).

```markdown
---
engine: flux2-klein          # flux2-klein | flux2-klein-fp8 | flux2-klein-fp4 | pulid-flux2-klein | pulid-flux1-dev | flux1-dev | flux1-schnell | sd35
width: 1024                  # pixels, must be a multiple of 64
height: 1024
steps: 50                    # inference steps (sd35 is fixed at 20)
guidance: 4.0                # CFG scale (sd35 is fixed at 1.0, flux1-schnell at 0.0)
seed: 42                     # optional — omit for a random result
negative_prompt: "blurry, low quality, watermark"
format: png                  # png | jpg | webp
face_image: /path/to/ref.png # pulid engines only — reference face (abs or relative to .md)
pulid_strength: 0.6          # pulid engines only — identity lock strength (default 0.6, 0.8 for flux1-dev)
---

Your prompt text here. Can span multiple paragraphs.
The entire body below the frontmatter is used as the prompt.
```

All frontmatter fields are optional. Missing values fall back to `imagecli.toml`, then to hardcoded defaults.

## User Config (`imagecli.toml`)

Optional global config file. imagecli walks upward from CWD to `$HOME` looking for `imagecli.toml`. Place it at `~/imagecli.toml` for a global default, or in a project parent directory to scope it narrowly.

```bash
cp imagecli.example.toml ~/imagecli.toml   # then edit to taste
```

```toml
[defaults]
engine     = "flux2-klein"       # default engine
width      = 1024                # output width in pixels
height     = 1024                # output height in pixels
steps      = 50                  # inference steps
guidance   = 4.0                 # CFG scale
output_dir = "images/images_out" # where to save generated images
format     = "png"               # png | jpg | webp
quality    = 95                  # JPEG/WebP quality (ignored for PNG)
```

**Priority:** CLI flag > `.md` frontmatter > `imagecli.toml` > hardcoded default

## Performance Optimizations

All optimizations are applied automatically when a model loads. The only manual choice is `--no-compile`.

### torch.compile

The transformer (or UNet) can be compiled with `torch.compile(mode="max-autotune-no-cudagraphs")` on first load, fusing CUDA kernels for a 20-40% speedup after a one-time warmup.

**Note:** For the default `flux2-klein` engine, torch.compile is blocked by the quanto QLinear contiguity patch. The `flux2-klein-fp8` engine (torchao) supports compile, but is already 40% slower than quanto's Marlin FP8 GEMM kernels even without compile. In practice, quanto's hardware-native FP8 matmul is faster than anything torch.compile can offer. Compile remains useful for non-Klein engines (`flux1-dev`, `flux1-schnell`, `sd35`).

```bash
imagecli generate "quick test" --no-compile  # skip compile for faster startup
```

### VAE tiling and slicing

VAE tiling (`enable_tiling()`) and slicing (`enable_slicing()`) are always enabled. Tiling decodes the latent in tiles instead of all at once, which reduces peak VRAM for large images. Slicing processes the batch dimension one element at a time.

### TF32 matmul precision

`torch.set_float32_matmul_precision("high")` is set globally at model load time. On Blackwell GPUs (RTX 5070 Ti and newer), this routes float32 matrix multiplications through tensor cores using TF32 precision, giving roughly a **10–15% speedup** with negligible quality impact.

### inference_mode

All generation runs inside `torch.inference_mode()`, which disables autograd tracking and reduces memory overhead compared to `torch.no_grad()`.

## Memory Safety

### Preflight check

Before loading a model, `preflight_check()` reads current GPU and system memory and aborts early with a clear message if resources are insufficient:

- **VRAM:** compares `torch.cuda.mem_get_info()` free VRAM against the engine's `vram_gb` requirement. If insufficient, it names the shortfall and suggests closing other GPU processes (e.g. `ollama`).
- **System RAM:** reads `/proc/meminfo` and requires at least 4GB free. Override with `IMAGECLI_MIN_FREE_RAM_GB=8` if needed.

### GPU memory cap

`torch.cuda.set_per_process_memory_fraction(0.85)` caps imagecli's VRAM usage at 85% of total, leaving headroom for the OS and other processes.

### Cleanup after generation

`engine.cleanup()` is called in a `finally` block after every generation. It runs `torch.cuda.empty_cache()` and `gc.collect()` to release cached allocations back to the system promptly.

## Project Structure

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
    flux2_klein.py        — FLUX.2-klein-4B engine (default, quanto FP8)
    flux2_klein_fp8.py    — FLUX.2-klein-4B torchao FP8 (40% slower than quanto, torch.compile compatible)
    flux2_klein_fp4.py    — FLUX.2-klein-4B NVFP4 (Blackwell FP4 via comfy-kitchen)
    pulid_flux2_klein.py  — FLUX.2-klein-4B + PuLID face identity lock (optional extra)
    pulid_flux1_dev.py    — FLUX.1-dev GGUF + PuLID v0.9.1 face identity lock
    flux1_dev.py          — FLUX.1-dev fp8 quantized engine
    flux1_schnell.py      — FLUX.1-schnell fp8 quantized engine
    sd35.py               — SD3.5 Large Turbo engine
```

Output files never overwrite existing ones. If `image.png` already exists, the next run saves `image_1.png`, then `image_2.png`, and so on.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, code conventions, and how to add a new engine. All PRs target `staging` and require `ruff` + `pytest` to pass.

## Documentation

| Doc | Description |
|-----|-------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

## Model Research (March 2026)

- **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** — Black Forest Labs, Jan 2026. 4B parameter model distilled from FLUX.2-dev. Three quantization options: quanto FP8 (default), official pre-quantized FP8, and NVFP4 (Blackwell-native).
- **FLUX.2-klein-4B torchao FP8** — torchao weight-only FP8 quantization (no quanto). Enables torch.compile but ~40% slower than quanto's Marlin FP8 GEMM kernels. Useful if you need compile or want to avoid quanto. Requires `uv sync --group fp8`.
- **[FLUX.2-klein-4B NVFP4](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4)** — Official BFL NVFP4 weights for Blackwell GPUs. ~2 GB transformer via comfy-kitchen FP4 kernel. Requires sm_120+ and CUDA 13.0 (cu130).
- **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)** — Black Forest Labs. Still competitive for detailed, complex prompts. Non-commercial license. fp8 quantized via optimum-quanto to fit in 10GB.
- **[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)** — Black Forest Labs. Apache 2.0, ungated. 4-step distilled model for fast iteration. fp8 quantized.
- **[SD3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)** — Stability AI. 20-step CFG-free generation. Strong for photorealistic subjects. T5 encoder quantized to int8.
- **FLUX.2-dev (32B)** — Not included: requires 24GB+ VRAM even in GGUF format.

## License

MIT
