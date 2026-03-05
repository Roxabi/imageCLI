# imageCLI

![Python](https://img.shields.io/badge/python-3.11--3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9)
![CUDA](https://img.shields.io/badge/CUDA-enabled-76B900?logo=nvidia&logoColor=white)
![version](https://img.shields.io/badge/version-0.1.0-22c55e)

Unified CLI for local image generation — FLUX.2-klein, FLUX.1-dev, FLUX.1-schnell, and SD3.5 Large Turbo backends via HuggingFace Diffusers.

## Requirements

- Python 3.11–3.12
- CUDA GPU (16GB VRAM recommended; 10GB minimum for fp8 engines)
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
git clone <repo-url> && cd imageCLI
uv sync
cp imagecli.example.toml ~/imagecli.toml   # optional — customize defaults
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

## Engines

| Engine | Model | VRAM | Steps | Speed | Notes |
|---|---|---|---|---|---|
| `flux2-klein` | FLUX.2-klein-4B | ~13GB | 50 | ~20s | **Default.** Best quality/VRAM ratio. BFL Nov 2025 |
| `flux1-dev` | FLUX.1-dev fp8 | ~10GB | 50 | ~30s | Maximum quality, longer prompts. fp8 via optimum-quanto |
| `flux1-schnell` | FLUX.1-schnell fp8 | ~10GB | 4 | ~5s | Fastest. Apache 2.0, ungated. fp8 via optimum-quanto |
| `sd35` | SD3.5 Large Turbo | ~14GB | 20 | ~8s | CFG-free, realistic photos. T5 encoder quantized to int8 |

Speed estimates are for 1024x1024 on an RTX 5070 Ti (16GB) after torch.compile warmup. First run is slower due to compilation.

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
| `--engine` | `-e` | Engine name (`flux2-klein`, `flux1-dev`, `flux1-schnell`, `sd35`) | `flux2-klein` |
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

Processes every `.md` file in the directory in sorted order. Engine instances are cached across files — the model loads once per engine, not once per image.

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--engine` | `-e` | Override engine for all files | per-file frontmatter |
| `--output-dir` | | Output directory for all images | `images/images_out` |
| `--no-compile` | | Skip torch.compile | off |

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
engine: flux2-klein          # flux2-klein | flux1-dev | flux1-schnell | sd35
width: 1024                  # pixels, must be a multiple of 64
height: 1024
steps: 50                    # inference steps (sd35 is fixed at 20)
guidance: 4.0                # CFG scale (sd35 is fixed at 1.0, flux1-schnell at 0.0)
seed: 42                     # optional — omit for a random result
negative_prompt: "blurry, low quality, watermark"
format: png                  # png | jpg | webp
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

The transformer (or UNet) is compiled with `torch.compile(mode="max-autotune-no-cudagraphs")` on first load. This fuses CUDA kernels and typically gives a **20–40% speedup** after a one-time warmup cost of ~60–90 seconds on the first image.

Subsequent images in the same session run at full compiled speed.

Use `--no-compile` to skip compilation when you only need one image and don't want to wait for warmup:

```bash
imagecli generate "quick test" --no-compile
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
    flux2_klein.py        — FLUX.2-klein-4B engine (default)
    flux1_dev.py          — FLUX.1-dev fp8 quantized engine
    flux1_schnell.py      — FLUX.1-schnell fp8 quantized engine
    sd35.py               — SD3.5 Large Turbo engine
```

Output files never overwrite existing ones. If `image.png` already exists, the next run saves `image_1.png`, then `image_2.png`, and so on.

## Documentation

| Doc | Description |
|-----|-------------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

## Model Research (March 2026)

- **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** — Black Forest Labs, Nov 2025. 4B parameter model distilled from FLUX.2-dev (32B). Best quality that fits in 16GB VRAM without quantization.
- **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)** — Black Forest Labs. Still competitive for detailed, complex prompts. Non-commercial license. fp8 quantized via optimum-quanto to fit in 10GB.
- **[FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)** — Black Forest Labs. Apache 2.0, ungated. 4-step distilled model for fast iteration. fp8 quantized.
- **[SD3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)** — Stability AI. 20-step CFG-free generation. Strong for photorealistic subjects. T5 encoder quantized to int8.
- **FLUX.2-dev (32B)** — Not included: requires 24GB+ VRAM even in GGUF format.

## License

MIT
