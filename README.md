# imageCLI

Unified CLI for local image generation on your RTX 5070 Ti (16GB). Three backends, one simple interface.

## Engines

| Engine | Model | VRAM | Speed | Best for |
|---|---|---|---|---|
| `flux2-klein` | FLUX.2-klein-4B | ~13GB | ~20s | Default — best quality/VRAM ratio |
| `flux1-dev` | FLUX.1-dev fp8 | ~10GB | ~30s | Maximum quality, longer prompts |
| `sd35` | SD3.5 Large Turbo | ~14GB | ~8s | Speed, realistic photos |

## Install

```bash
cd imageCLI
uv sync
cp imagecli.example.toml ~/imagecli.toml   # optional, customize defaults
```

> Models are downloaded automatically on first use from HuggingFace (~13GB for FLUX.2-klein).

## Usage

```bash
# Quick generation
uv run imagecli generate "a cat in space"

# From a .md prompt file
uv run imagecli generate images/prompts_in/lyra-v1-cinematic.md

# Batch all prompts in a directory
uv run imagecli batch images/prompts_in/

# List engines
uv run imagecli engines

# Show GPU + config info
uv run imagecli info
```

## Prompt files

Write prompts as Markdown with YAML frontmatter:

```markdown
---
engine: flux2-klein
width: 1024
height: 1024
steps: 50
guidance: 4.0
seed: 42
negative_prompt: "blurry, low quality"
---

Your prompt here.
```

## Model Research (March 2026)

Best open-source image generation models:

- **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** — Black Forest Labs, Nov 2025. 4B param distilled from FLUX.2-dev 32B. Best quality that fits in 16GB.
- **[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)** — Still competitive. fp8 quantized via optimum-quanto for 16GB.
- **[SD3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)** — Stability AI. Fast 20-step CFG-free generation.
- **FLUX.2-dev** (32B) — Not included: needs 24GB+ VRAM even in GGUF.
