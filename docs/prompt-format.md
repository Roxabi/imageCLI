# Prompt Format

Markdown prompts with YAML frontmatter.

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

Priority: CLI flag > frontmatter > `imagecli.toml` > default.
