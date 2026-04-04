# LoRA Training & Inference

LoRA training is handled by external tools. LoRA inference is supported on `flux2-klein` and `flux2-klein-fp8` engines.

---

## Training (external — ai-toolkit)

Use [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit/) for LoRA training on Klein 4B. Day-0 FLUX.2 support, 20-30% faster than SimpleTuner, battle-tested with 50+ published Klein training runs.

Other viable tools: diffusers `train_dreambooth_lora_flux2_klein.py`, SimpleTuner, DiffSynth-Studio. kohya_ss/sd-scripts Klein 4B support is unconfirmed.

### Critical rules

- Always train on `black-forest-labs/FLUX.2-klein-base-4B` (undistilled). **Never** train on the distilled `FLUX.2-klein-4B` — distilled models resist fine-tuning (#1 cause of "my LoRA doesn't work" in the FLUX community).
- Training data must come from **Klein 4B native** (no PuLID, no LoRA). PuLID-generated data produces smooth/plastic LoRA output.
- 25-30 curated images + 20-30 regularization images. 200+ causes overfitting + style dilution.

### Recommended config (RTX 5070 Ti, 16 GB)

| Parameter | Value | Notes |
|-----------|-------|-------|
| rank | 16 (start), 32-64 for stronger identity | Higher rank = LoRA can fight longer prompts |
| alpha | equal to rank | alpha/rank = effective weight |
| lr | 1e-4 (adamw8bit) or 1.0 (Prodigy) | >2e-4 causes instant divergence |
| steps | 2000-2500 | Monitor checkpoints every 250 |
| batch_size | 1 | Mandatory at 16 GB |
| resolution | 768 (multi-bucket [512, 768, 1024]) | 512-only produces blurry output at 1024 inference |
| gradient_checkpointing | true | Required for 16 GB |
| cache_latents | true | Frees VAE during training |
| quantize | true | FP8 (native on sm_120) |

### Training data

| Aspect | Recommendation |
|--------|---------------|
| **Source** | Klein 4B native (no PuLID, no LoRA), cherry-picked for face similarity |
| **Count** | 25-30 curated + 20-30 regularization |
| **Resolution** | Multi-bucket: 512, 768, 1024 |
| **Diversity** | 40% frontal, 30% three-quarter, 20% profile, 10% other. 6+ outfits, 5+ backgrounds |
| **Captions** | Natural language with trigger word. Describe everything EXCEPT the face |
| **Regularization** | 20-30 generic images without trigger word (prevents face leaking into all generations) |

### Known pitfalls

| Pitfall | Mitigation |
|---------|------------|
| Training on distilled model | Always use `FLUX.2-klein-base-4B` |
| LR too high (>2e-4) | Start at 1e-4 or use Prodigy |
| No regularization images | Include 20-30 generic images without trigger word |
| 512-only training | Multi-resolution bucket training |
| Too many steps (>3000) | Monitor checkpoints every 250; weight decay 0.01-0.1 |
| PuLID/FLUX.1-dev training data | Use Klein 4B native images only |

---

## LoRA Inference

Supported on `flux2-klein` and `flux2-klein-fp8`. Not supported on `flux2-klein-fp4` (pre-quantized weights).

### Usage

```bash
# CLI flag
imagecli generate "lyraface person smiling, studio lighting" --lora ./lora.safetensors
imagecli generate "lyraface person" --lora ./lora.safetensors --lora-scale 1.5
imagecli batch prompts/ --lora ./lora.safetensors

# Or via frontmatter
# ---
# lora_path: /path/to/lora.safetensors
# lora_scale: 1.5
# ---
```

CLI flags override frontmatter values.

### How it works

LoRA weights are fused into the base bf16 weights before FP8 quantization:

1. Load pipeline (bf16)
2. `load_lora_weights(path)` — adds PEFT adapter
3. `set_adapters(scale)` — if scale != 1.0
4. `fuse_lora()` — merges adapter into base weights
5. `unload_lora_weights()` — removes adapter structure
6. Quantize fused weights to FP8
7. Move to GPU

This means LoRA is baked in at load time. Changing the LoRA requires reloading the engine.

### Load order constraint

| Operation order | Works? | Notes |
|---|---|---|
| Load LoRA → quantize (FP8) → GPU | Yes | LoRA baked into weights before quantization |
| Quantize → GPU → load LoRA | **No** | LoRA silently applies but has no effect on quantized weights |
| No quantize, CPU offload, swap LoRA freely | Yes | Slower (~12s/img) but correct for checkpoint comparison |

### Prompt discipline with LoRA

The LoRA competes with the prompt for control of the output. At rank 16:

| Prompt style | Result |
|---|---|
| `"lyraface person smiling, studio lighting"` | Correct face — prompt is vague, LoRA fills in identity |
| `"lyraface person. Young woman, mid-twenties, dark blonde hair..."` | Wrong face — prompt overrides the LoRA |

**Rule:** Keep prompts short when using low-rank LoRA. Trigger word + scene/mood/lighting. Don't describe the face. Increase rank to 32-64 if longer prompts are needed.

### Tuning levers

| Parameter | Set at | Effect of increasing |
|---|---|---|
| **rank** | Training | More capacity. 32-64 can fight longer prompts |
| **alpha** | Training | alpha/rank = effective weight |
| **training steps** | Training | Stronger lock, but overfitting risk |
| **adapter_weights** | Inference | Boost LoRA scale without retraining |
| **inference steps** | Inference | Better quality, diminishing returns past 20 |

---

## Sources

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit/) — Klein 4B/9B LoRA training
- [diffusers Klein training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py)
- [BFL Klein Training Docs](https://docs.bfl.ai/flux_2/flux2_klein_training)
- [FLUX.2-klein-base-4B on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- [50+ Klein LoRA Training Runs](https://medium.com/@calvinherbst/50-flux-2-klein-lora-training-runs-dev-and-klein-to-see-what-config-parameters-actually-matter-3196e4f64fd5)
- [SimpleTuner FLUX.2 Quickstart](http://docs.simpletuner.io/quickstart/FLUX2/)
