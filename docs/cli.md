# CLI Reference

## Single generation

```bash
imagecli generate "text" [-e ENGINE] [-W W -H H] [--steps N] [--guidance F] \
  [--seed N] [-n "neg"] [-o OUT] [--output-dir DIR] [--no-compile] \
  [--lora PATH] [--lora-scale F]

imagecli generate prompt.md [-e ENGINE]             # from .md with frontmatter
```

Common flags:

| Flag | Default | Notes |
|---|---|---|
| `-e ENGINE` | config | Override engine selection |
| `-W W -H H` | 1024×1024 | Must be multiples of 64 |
| `--steps N` | 50 (sd35: 20, schnell: 4) | Inference steps |
| `--guidance F` | 4.0 (sd35: 1.0, schnell: 0.0) | CFG scale |
| `--seed N` | random | Reproducibility |
| `-n "neg"` | — | Negative prompt |
| `-o OUT` | auto | Custom output path |
| `--output-dir DIR` | `images/images_out/` | Custom output directory |
| `--no-compile` | compile on | Skip `torch.compile` (faster startup) |
| `--lora PATH` | — | LoRA weights (flux2-klein / -fp8) |
| `--lora-scale F` | 1.0 | Adapter scale (try 1.5 for stronger identity) |

## Batch (all .md in a directory)

```bash
imagecli batch DIR [-e ENGINE] [--output-dir DIR] [--no-compile] \
  [--two-phase] [--lora PATH]
```

- `--two-phase` — lower VRAM mode (encode → generate in two passes)
- See `docs/performance.md` for when to use it.

## Info

```bash
imagecli engines     # list engines with VRAM requirements
imagecli info        # show active config + GPU info
```

## Examples

```bash
# Single generation
imagecli generate "a cat in space"
imagecli generate "a cat in space" -e flux1-dev
imagecli generate prompt.md                         # from .md with frontmatter
imagecli generate prompt.md -e flux1-dev            # override engine
imagecli generate "prompt" -W 1280 -H 720           # custom size
imagecli generate "prompt" --steps 30 --guidance 4.0
imagecli generate "prompt" --seed 42                # reproducible
imagecli generate "prompt" -n "blurry, ugly"        # negative prompt
imagecli generate "prompt" -o my_image.png          # custom output path
imagecli generate "prompt" --output-dir ./out       # custom output dir
imagecli generate "prompt" --no-compile             # skip torch.compile
imagecli generate "prompt" --lora ./lora.safetensors
imagecli generate "prompt" --lora ./lora.safetensors --lora-scale 1.5

# Batch generation
imagecli batch images/prompts_in/
imagecli batch images/prompts_in/ -e flux1-dev
imagecli batch images/prompts_in/ --output-dir ./results
imagecli batch images/prompts_in/ --no-compile
imagecli batch images/prompts_in/ --two-phase       # force 2-phase (lower VRAM)
imagecli batch images/prompts_in/ --lora ./lora.safetensors
```
