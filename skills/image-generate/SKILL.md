---
name: image-generate
description: 'Generate images with imageCLI — select the right engine, write prompts, handle single or batch generation with VRAM-aware defaults. Triggers: "generate image" | "generate an image" | "image generate" | "create image" | "make an image" | "imagecli generate" | "imagecli batch" | "batch images" | "generate avatar" | "face lock image" | "/image-generate".'
version: 0.1.0
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion
---

# Image Generate

Generate images using imageCLI on the local GPU. This skill selects the right engine, writes prompt files, and runs generation.

**imageCLI location:** `~/projects/imageCLI`
**Default output:** `images/images_out/`
**Prompt files:** `images/prompts_in/` (tracked in git)

## Engine Selection Guide

| Use case | Engine | Why |
|----------|--------|-----|
| General image generation | `flux2-klein` (default) | Best quality-to-VRAM ratio, 16GB GPUs |
| Maximum quality, complex prompts | `flux1-dev` | Better at long detailed prompts, text rendering |
| Quick iteration, drafts | `flux1-schnell` | 4-step, ~5s per image, Apache 2.0 |
| Photorealistic subjects | `sd35` | CFG-free, strong on realistic photos |
| Face-locked images (recommended) | `pulid-flux1-dev` | Clean face lock, no banding, 1024x1024 |
| Face-locked images (Klein quality) | `pulid-flux2-klein` | Klein texture + face lock via dim projection |
| LoRA-based identity | `flux2-klein` + `--lora` | Pre-trained LoRA, no PuLID deps, natural texture |

### Performance: Single vs Batch

- **1-2 images:** Use `imagecli generate`. Klein uses CPU offload (~8 GB, no compile warmup, ~25s per image).
- **3+ images:** Use `imagecli batch`. Klein uses 2-phase (encode all prompts then generate all images, compiled, ~14s per image after warmup).
- **`--no-compile`:** Use for one-off generation when you want fast startup (~25s) instead of waiting for compile (~90s warmup + ~14s per image).

## Phase 1 — Understand the Request

Determine from the user's request:
1. **Subject/concept** — what to generate
2. **Quantity** — single image or batch
3. **Face lock needed?** — does the user want a specific person's face
4. **Quality vs speed** — quick draft or final quality
5. **Output location** — default or custom

If unclear, ask with `AskUserQuestion`:
```
What would you like to generate?

- Subject: (describe the image)
- Quantity: 1 image / batch of N
- Face reference: (path to face photo, or "no")
- Engine preference: auto / flux2-klein / flux1-dev / flux1-schnell / sd35
- Output: default (images/images_out/) or custom path
```

## Phase 2 — Select Engine

Based on Phase 1:
- Face lock requested (reference image) -> `pulid-flux1-dev` (or `pulid-flux2-klein` if user prefers Klein texture)
- Face lock requested (trained LoRA) -> `flux2-klein --lora path/to/lora.safetensors`
- Batch of 3+ without face lock -> `flux2-klein` (2-phase kicks in automatically)
- Quick draft -> `flux1-schnell`
- Complex detailed prompt -> `flux1-dev`
- Photorealistic -> `sd35`
- Default -> `flux2-klein`

## Phase 3 — Write Prompt

### Single image (inline prompt)

```bash
cd ~/projects/imageCLI
uv run imagecli generate "your prompt here" -e <engine> --seed 42
```

### Single image from .md file

Write a prompt file:

```markdown
---
engine: flux2-klein
width: 1024
height: 1024
steps: 50
guidance: 4.0
seed: 42
---

Detailed prompt text here. Can span multiple paragraphs.
Use vivid, specific language. Describe lighting, composition, style.
```

For face-locked images (PuLID), add:
```yaml
face_image: /absolute/path/to/reference.png
pulid_strength: 0.6    # 0.4-0.8, higher = stronger face lock
```

For LoRA-based identity (flux2-klein, flux2-klein-fp8), add:
```yaml
lora_path: /path/to/lora.safetensors
lora_scale: 1.0    # try 1.5 for stronger identity
```

Then generate:
```bash
cd ~/projects/imageCLI
uv run imagecli generate path/to/prompt.md
```

### Batch generation

1. Write multiple `.md` prompt files to a directory
2. Run batch:

```bash
cd ~/projects/imageCLI
uv run imagecli batch path/to/prompts/ -e flux2-klein
uv run imagecli batch path/to/prompts/ -e flux2-klein --output-dir ./results
```

For Klein batches (3+ files), 2-phase mode activates automatically:
- Phase 1: Encodes all prompts (~8 GB VRAM, fast)
- Phase 2: Generates all images (~4 GB VRAM, compiled, fast)

## Phase 4 — Run Generation

Execute the command. After generation:
1. Report the output path
2. Report peak VRAM from the output
3. If batch, report success/failure counts

## Prompt Writing Tips

- Be specific: "soft golden hour light from the left" > "good lighting"
- Describe composition: "close-up portrait", "wide establishing shot", "isometric view"
- Include style cues: "cinematic", "studio photograph", "digital illustration", "watercolor"
- For Klein: prompts can be shorter and still produce good results (4B model is less sensitive to prompt length)
- For flux1-dev: longer, more detailed prompts work better
- flux1-schnell ignores guidance (always 0.0) and uses 4 steps by default
- sd35 ignores guidance (always 1.0) and uses 20 steps

## Error Handling

- **VRAM errors:** Close other GPU processes (ollama, ComfyUI). Run `nvidia-smi` to check.
- **Model not found:** First run downloads models from HuggingFace (~13 GB for Klein). Ensure `HF_TOKEN` is set for gated models.
- **PuLID engines fail:** Need `uv sync --extra pulid` and model weights at `~/ComfyUI/models/pulid/`.
- **LoRA has no effect:** LoRA only works on `flux2-klein` and `flux2-klein-fp8`. Not supported on `flux2-klein-fp4` (pre-quantized weights).

## Quick Reference

```bash
# Check GPU and config
uv run imagecli info
uv run imagecli engines

# Single image
uv run imagecli generate "prompt" -e flux2-klein
uv run imagecli generate "prompt" -e flux1-schnell  # fast draft
uv run imagecli generate prompt.md                   # from file

# Batch
uv run imagecli batch prompts_dir/ -e flux2-klein
uv run imagecli batch prompts_dir/ --no-compile      # skip compile for small batches

# LoRA generation
uv run imagecli generate "lyraface person smiling" --lora ./lora.safetensors
uv run imagecli batch prompts_dir/ --lora ./lora.safetensors --lora-scale 1.5
```

All commands must be run from `~/projects/imageCLI` (or wherever imageCLI is installed).
