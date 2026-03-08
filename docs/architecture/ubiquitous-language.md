# Ubiquitous Language

Domain glossary for imageCLI. Use these terms consistently in code, docs, commit messages,
and conversations.

## Core Concepts

| Term | Definition | Where in code |
|---|---|---|
| **Engine** | A named image generation backend wrapping a specific Diffusers pipeline. Identified by a kebab-case key (e.g. `flux1-dev`). | `engine.py:ImageEngine` |
| **Port** | The abstract interface (`ImageEngine` ABC) that all engines implement. Defines `_load()`, `generate()`, `cleanup()`. | `engine.py:ImageEngine` |
| **Adapter** | A concrete engine class that implements the port using a specific model (e.g. `Flux1DevEngine`). | `engines/*.py` |
| **Registry** | The dict mapping engine names to engine classes. Single source of truth for which engines exist. | `engine.py:_get_registry()` |
| **Pipeline** / **Pipe** | The loaded Diffusers model object (`self._pipe`). Held as engine state after `_load()`. | `engine._pipe` |
| **Prompt** | User-provided text description of the desired image. Can come from CLI args, `.md` files, or library kwargs. | `generate(prompt=...)` |
| **PromptDoc** | Parsed representation of a `.md` prompt file — prompt text plus optional frontmatter fields. | `markdown.py:PromptDoc` |
| **Frontmatter** | YAML metadata block at the top of a `.md` prompt file (between `---` markers). Overrides config defaults. | `markdown.py:parse_prompt_file()` |

## Generation Parameters

| Term | Definition | Notes |
|---|---|---|
| **Steps** | Number of denoising iterations during inference. More steps = higher quality, slower. | Some engines have fixed steps (SD3.5 Turbo: 20, Schnell: 4) |
| **Guidance** / **CFG scale** | Classifier-Free Guidance scale. Higher = closer to prompt, lower = more creative. | Some engines ignore this (Schnell: 0.0, SD3.5: 1.0) |
| **Negative prompt** | Text describing what to avoid in the image. | Not supported by FLUX engines |
| **Seed** | Random number generator seed for reproducible results. `None` = non-deterministic. | `torch.Generator("cuda").manual_seed(seed)` |
| **Format** | Output image format: `png`, `jpg`, or `webp`. | Resolved by priority chain |

## Engine Capabilities

| Term | Definition | Where in code |
|---|---|---|
| **EngineCapabilities** | Dataclass declaring which parameters an engine actually uses vs ignores. | `engine.py:EngineCapabilities` |
| **fixed_steps** | When not `None`, the engine always runs this many steps regardless of user input. | `EngineCapabilities.fixed_steps` |
| **fixed_guidance** | When not `None`, the engine always uses this CFG scale. | `EngineCapabilities.fixed_guidance` |
| **negative_prompt** (capability) | Whether the engine supports negative prompts. `False` for FLUX engines. | `EngineCapabilities.negative_prompt` |

## GPU and Memory

| Term | Definition | Where in code |
|---|---|---|
| **Compute capability** | CUDA version identifier as `(major, minor)` tuple. Determines quantization type. | `engine.py:get_compute_capability()` |
| **sm** | Streaming Multiprocessor version (e.g. `sm_120` = Blackwell). Shorthand for compute capability. | Used in quantization selection |
| **Preflight check** | Resource validation before model loading. Checks VRAM and system RAM. Fails fast. | `engine.py:preflight_check()` |
| **InsufficientResourcesError** | Exception raised when preflight check fails. Subclasses `RuntimeError`. | `engine.py:InsufficientResourcesError` |
| **CPU offload** | Diffusers feature that streams model layers to GPU one at a time, reducing peak VRAM. | `pipe.enable_model_cpu_offload()` |
| **VRAM cap** | Limits PyTorch to 85% of GPU memory, leaving headroom for CUDA runtime. | `torch.cuda.set_per_process_memory_fraction(0.85)` |
| **Cleanup** | Free GPU memory after generation: `torch.cuda.empty_cache()` + `gc.collect()`. | `ImageEngine.cleanup()` |

## Quantization

| Term | Definition | When used |
|---|---|---|
| **fp8** (`qfloat8`) | 8-bit floating point quantization. Native hardware support on sm >= (8,9). | FLUX engines on Ada Lovelace / Blackwell |
| **int8** (`qint8`) | 8-bit integer quantization. Works on Ampere (sm >= 8,0) and newer. | FLUX engines on Ampere; SD3.5 T5 encoder always |
| **bf16** | Brain float 16-bit. Default when quantization is not applied or fails. | FLUX.2-klein always; fallback for all engines |
| **Freeze** | Locks quantized weights so they are not modified during inference. Must follow `quantize()`. | `optimum.quanto.freeze()` |

## Performance

| Term | Definition | Where in code |
|---|---|---|
| **torch.compile** | JIT kernel fusion for 20-40% inference speedup after first-run warmup. | `_optimize_pipe()` |
| **TF32 matmul** | Tensor Float 32 precision for matrix multiplication. ~10-15% speedup on Blackwell. | `torch.set_float32_matmul_precision("high")` |
| **VAE tiling** | Processes VAE decode in tiles to reduce peak VRAM for large images. | `pipe.vae.enable_tiling()` |
| **VAE slicing** | Processes VAE decode in slices for batched generation. | `pipe.vae.enable_slicing()` |
| **`--no-compile`** | CLI flag to skip `torch.compile`. Faster startup, slower inference. | `cli.py`, `engine._compile` |

## Configuration

| Term | Definition | Where in code |
|---|---|---|
| **Priority chain** | Resolution order: CLI flag > .md frontmatter > imagecli.toml > hardcoded default. | `cli.py:generate()`, `__init__.py:generate()` |
| **Config walking** | `imagecli.toml` is found by walking up from CWD to `$HOME`. First match wins. | `config.py:_find_config()` |
| **Anti-clobber** | Output files are never overwritten. Suffix `_1`, `_2`, etc. appended automatically. | `_utils.py:resolve_output()` |

## Lifecycle States

An engine instance transitions through these states:

```
Unloaded ──_load()──► Loaded ──generate()──► Loaded ──cleanup()──► Cleaned
   │                    │                       │                      │
   │ _pipe = None       │ _pipe = Pipeline      │ (reusable)          │ VRAM freed
   │ no GPU memory      │ model on GPU/CPU      │                     │ GC collected
```

- **Unloaded**: After `__init__()`. No GPU memory consumed. Safe to hold many instances.
- **Loaded**: After first `generate()` call triggers `_load()`. Model weights in memory.
- **Cleaned**: After `cleanup()`. VRAM cache cleared, Python GC triggered. Instance is
  no longer usable for generation without re-loading.

## Common Confusions

| Often confused | Distinction |
|---|---|
| Engine vs Pipeline | Engine is the imagecli wrapper; pipeline is the Diffusers object inside `self._pipe` |
| Steps (requested) vs Steps (effective) | User requests 30 steps, but SD3.5 Turbo always runs 20. `effective_steps()` returns the real count. |
| `generate()` on engine vs `generate()` on facade | `engine.generate()` runs inference; `imagecli.generate()` handles the full workflow (config → preflight → generate → cleanup) |
| Preflight check vs cleanup | Preflight runs **before** loading (fail fast). Cleanup runs **after** generation (free resources). |
| VRAM cap vs preflight | VRAM cap (`0.85`) limits PyTorch's allocator. Preflight checks **free** VRAM against `engine.vram_gb`. |
