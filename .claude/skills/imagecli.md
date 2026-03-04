# /imagecli

Generate images locally using the imagecli CLI.

## Quick reference

```bash
# Single generation (default engine: flux2-klein)
uv run imagecli generate "your prompt here"

# Choose engine
uv run imagecli generate "your prompt" -e flux1-dev
uv run imagecli generate "your prompt" -e sd35

# Custom resolution
uv run imagecli generate "your prompt" -W 1280 -H 720

# Reproducible (fixed seed)
uv run imagecli generate "your prompt" --seed 42

# From a .md prompt file
uv run imagecli generate images/prompts_in/lyra-v1-cinematic.md

# Batch all .md files in a directory
uv run imagecli batch images/prompts_in/

# Show engines + VRAM + defaults
uv run imagecli engines

# Show active config + GPU
uv run imagecli info
```

## Engine defaults (tuned per model)

| Engine | Steps | Guidance | VRAM |
|--------|-------|----------|------|
| `flux2-klein` | 28 | 3.5 | ~13GB |
| `flux1-dev` | 50 | 3.5 | ~10GB |
| `sd35` | 20 | 1.0 | ~14GB |

## Prompt file format

Create `.md` files in `images/prompts_in/` with YAML frontmatter:

```markdown
---
engine: flux2-klein
width: 1024
height: 1024
steps: 28
guidance: 3.5
seed: 42
negative_prompt: "blurry, low quality"
---

Your prompt text here.
```

## Config override priority

CLI flag > `.md` frontmatter > `~/imagecli.toml` > engine defaults

## When asked to generate an image

1. Ask for the prompt (or use the one provided)
2. Ask for engine preference if not specified (default: flux2-klein)
3. Check if a `.md` prompt file already exists in `images/prompts_in/`
4. If not, create one with appropriate frontmatter
5. Run `uv run imagecli generate images/prompts_in/<name>.md`
6. Report the output path

## When asked to batch generate

Run: `uv run imagecli batch images/prompts_in/`

Output images go to `images/images_out/` by default.
