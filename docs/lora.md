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

Supported on `flux2-klein`, `flux2-klein-fp8`, and `flux2-klein-fp4` (via runtime NVFP4 quantization of the fused bf16 base).

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

## Multi-LoRA Stacking

Stack multiple LoRA adapters in a single generation. All adapters are fused into the base bf16 weights before FP8 quantization — same load-time bake as single-LoRA.

### Frontmatter example

```yaml
---
engine: flux2-klein
loras:
  - path: /path/to/subject1.safetensors
    trigger: lyraface
    scale: 1.0
  - path: /path/to/style.safetensors
    scale: 0.8
---

lyraface person in impressionist painting style, warm afternoon light
```

Each entry maps to a `LoraSpec(path, scale, trigger, embedding_path)`. `scale` and `trigger` are optional per-adapter.

### CLI example

Repeat `--lora` for each adapter. Paired flags (`--lora-scale`, `--trigger`, `--embedding-path`) are matched positionally and must appear the same number of times as `--lora`, or be omitted entirely for defaults.

```bash
imagecli generate prompt.md \
    --lora /path/to/subject1.safetensors --trigger lyraface \
    --lora /path/to/style.safetensors --lora-scale 0.8
```

### Override policy

When any `--lora*` CLI flag is set, the CLI list fully replaces the frontmatter `loras:` list (no merge). Omit CLI LoRA flags to use the frontmatter list as-is.

### Pivotal atomicity

When multiple LoRAs carry pivotal embeddings, all trigger tokens and trained vectors are applied in a single atomic operation: trigger-collision or vocab-collision with any LoRA's placeholders aborts before any tokenizer mutation, so either every pivotal lands or none do.

### Engine support

| Engine | Multi-LoRA | Notes |
|---|---|---|
| `flux2-klein` | Yes | quanto FP8; adapters fused pre-quantization |
| `flux2-klein-fp8` | Yes | torchao FP8; same fuse order |
| `flux2-klein-fp4` | **No** | NVFP4 is pre-quantized; LoRA fuse into a frozen quantized checkpoint is unsupported. Raises `ValueError` if `loras` is non-empty. |

### Practical stacking depth

No hard cap is enforced on `N`, but stacking more than 2–3 adapters is untested and not recommended. Each additional adapter adds roughly 0.5–1 GB VRAM at fuse time, and identity fidelity tends to degrade as trained directions in weight space conflict and cancel. Start with N=2 and measure before going higher.

### Mixed-form error

Setting both `loras:` and any singular key (`lora_path`, `lora_scale`, `trigger`, `embedding_path`) in the same source raises `ValueError`. This applies to frontmatter, engine constructor kwargs, and NATS payload. Use one form or the other — `loras:` list for multi-LoRA, singular keys for legacy single-LoRA.

---

## Pivotal Tuning Inference

When ai-toolkit is configured with both a `network:` (LoRA) block AND an `embedding:` (pivotal tuning) block, it trains N placeholder token embeddings alongside the transformer LoRA. Those embeddings tighten the trigger-word semantics in the text encoder — the 16 GB-feasible alternative to full TE training, which OOMs on Klein 4B's Qwen3 TE.

**imageCLI supports pivotal tuning inference on `flux2-klein`, `flux2-klein-fp4`, and `flux2-klein-fp8`.**

### Why it needs a flag (the silent-drop failure mode)

Before this feature, passing a pivotal-trained LoRA to `imagecli generate --lora X.safetensors` would load the transformer delta correctly but silently drop the trained `emb_params` tensor. The Qwen3 tokenizer never heard of the trigger, BPE-split it into sub-tokens with default embeddings, and the TE-side training was wasted. Training looked fine, inference ran without errors, generations looked acceptable — and pivotal-trained LoRAs scored identically to non-pivotal ones.

imageCLI now **hard-errors** when a LoRA contains `emb_params` but no trigger was resolved. Passing `--trigger <word>` is required.

### Usage

```bash
# CLI flag — merged format (emb_params inside the LoRA safetensors)
imagecli generate "lyraface cat on a bench" --lora ./v23b_000002000.safetensors --trigger lyraface

# Standalone embedding file (ai-toolkit A1111-format sibling)
imagecli generate "lyraface cat" \
    --lora ./v23b_lora_000002000.safetensors \
    --embedding ./lyraface000002000.safetensors \
    --trigger lyraface

# Batch
imagecli batch prompts/ --lora ./v23b.safetensors --trigger lyraface

# Or via frontmatter (CLI flag overrides)
# ---
# lora_path: /path/to/v23b_000002000.safetensors
# trigger: lyraface
# ---
```

### User rule: write the bare trigger once

The prompt expansion is **not idempotent**. imageCLI rewrites `"lyraface cat"` into `"lyraface lyraface_1 lyraface_2 lyraface_3 cat"` before tokenization, matching the diffusers textual-inversion expansion format. If you manually pre-expand (`"lyraface lyraface_1 cat"`), imageCLI logs a warning and skips expansion — but it's simpler to just write the bare trigger once and let the pipeline handle it.

### How it works

1. Load pipeline (bf16) — tokenizer + text encoder + transformer
2. Load LoRA → fuse → unload (unchanged)
3. **Pivotal hook** — runs before transformer quantization, while the TE is still on CPU in bf16:
   - Read `emb_params` tensor from the LoRA safetensors (merged) or standalone file
   - Validate shape: `(N, 2560)` with `1 <= N <= 32`
   - Add placeholder tokens `[trigger, trigger_1, ..., trigger_{N-1}]` to the Qwen3 tokenizer
   - `text_encoder.resize_token_embeddings(len(tokenizer))`
   - Write the N trained vectors into `embed_tokens.weight[new_ids]`
   - **Deterministic round-trip assertion** — reads back the vectors and asserts they match (`atol=5e-2`, bf16 precision bound)
   - Monkey-patch `pipe.encode_prompt` to run `_maybe_convert_prompt` before delegating
4. Quantize transformer (unchanged — only `nn.Linear` layers; the `nn.Embedding` in the TE is untouched)
5. Generate — user's prompt is rewritten inside the patched `encode_prompt` before tokenization

The tokenizer and TE changes are **disjoint from LoRA fuse/unload and from transformer quantization**. `unload_lora_weights()` only touches PEFT adapters on the transformer; `quantize(transformer, ...)` only touches `nn.Linear`. The `nn.Embedding` that holds the placeholder vectors survives both.

### Supported formats

| Format | Location | Trigger source |
|---|---|---|
| **Merged** | `emb_params` key inside the LoRA safetensors (ai-toolkit writes this via `extra_state_dict`) | `--trigger` or frontmatter `trigger:` (required) |
| **Standalone** | Separate `{trigger}{step}.safetensors` with `emb_params` tensor + metadata `string_to_param = {"*": "emb_params"}` + metadata `name` | `--trigger`, frontmatter `trigger:`, or auto-inferred from metadata `name` |

### Error cases

| Condition | Behavior |
|---|---|
| LoRA has `emb_params`, no trigger resolved | **Hard error** — passes `--trigger` message; prevents silent drop |
| `emb_params.shape[-1] != 2560` | **Hard error** — "LoRA trained against a different base model" |
| `emb_params.ndim != 2` or `N < 1` or `N > 32` | **Hard error** with shape in the message |
| `--trigger` provided but no `emb_params` found | **Warning + continue** (degraded to vanilla LoRA) |
| Trigger word already exists in Qwen3 vocab | **Hard error** — use a different trigger |
| Prompt already contains `{trigger}_1` (manual pre-expansion) | **Warning** on first inference — prompt is passed through unchanged |

---

## Sources

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit/) — Klein 4B/9B LoRA training
- [diffusers Klein training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py)
- [BFL Klein Training Docs](https://docs.bfl.ai/flux_2/flux2_klein_training)
- [FLUX.2-klein-base-4B on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- [50+ Klein LoRA Training Runs](https://medium.com/@calvinherbst/50-flux-2-klein-lora-training-runs-dev-and-klein-to-see-what-config-parameters-actually-matter-3196e4f64fd5)
- [SimpleTuner FLUX.2 Quickstart](http://docs.simpletuner.io/quickstart/FLUX2/)
