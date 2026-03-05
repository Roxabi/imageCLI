# Engine Architecture

## Overview

Each image backend is a subclass of `ImageEngine` (defined in `src/imagecli/engine.py`).
Engines are stateful objects: they hold a reference to the loaded pipeline and are
reused across repeated calls within a single CLI invocation.

## Engine Lifecycle

```
__init__()
  └── self._pipe = None          (no model, no GPU memory consumed)

preflight_check(engine)          (called by CLI before first generate)
  ├── check free VRAM >= engine.vram_gb
  └── check free system RAM >= MIN_FREE_RAM_GB

generate(prompt, ..., output_path)
  ├── _load()                    (idempotent — returns early if already loaded)
  │     ├── import torch, diffusers
  │     ├── Pipeline.from_pretrained(model_id, torch_dtype=bfloat16)
  │     ├── quantize(transformer, weights=qfloat8)  [fp8 engines only]
  │     ├── freeze(transformer)                     [fp8 engines only]
  │     ├── pipe.enable_model_cpu_offload(gpu_id=0)
  │     ├── torch.cuda.set_per_process_memory_fraction(0.85)
  │     └── _optimize_pipe(pipe)
  ├── build Generator from seed (optional)
  ├── torch.inference_mode()
  ├── pipe(prompt, width, height, steps, guidance, generator)
  └── image.save(output_path)

cleanup()                        (called by CLI in finally block)
  ├── torch.cuda.empty_cache()
  └── gc.collect()
```

## Base Class: `ImageEngine`

Defined in `src/imagecli/engine.py`.

**Abstract method:** `generate()` — every subclass must implement it.

**Concrete helpers:**

- `_optimize_pipe(pipe, compile=True)` — applies TF32 matmul precision, VAE tiling/slicing, and `torch.compile` on the transformer or unet. Compile mode is `max-autotune-no-cudagraphs` (safe for variable input shapes). Compilation is skipped after the first call (`_compiled` flag).
- `cleanup()` — frees VRAM cache and triggers Python GC.

**Class attributes every engine must set:**

| Attribute | Type | Purpose |
|---|---|---|
| `name` | `str` | Registry key (kebab-case, e.g. `flux2-klein`) |
| `description` | `str` | Shown by `imagecli engines` |
| `model_id` | `str` | HuggingFace repo ID |
| `vram_gb` | `float` | Minimum free VRAM required (used by preflight check) |

## Registry

`engine.py:_get_registry()` returns a dict mapping engine names to classes:

```python
{
    "flux2-klein": Flux2KleinEngine,
    "flux1-dev": Flux1DevEngine,
    "flux1-schnell": Flux1SchnellEngine,
    "sd35": SD35Engine,
}
```

`get_engine(name, compile=True)` instantiates and returns an engine from the registry.
`list_engines()` iterates class attributes without instantiating, keeping it fast.

## Adding a New Engine

1. Create `src/imagecli/engines/<slug>.py`.
2. Subclass `ImageEngine`, set the four class attributes.
3. Implement `__init__` (set `self._pipe = None`), `_load()`, and `generate()`.
4. In `_load()`: load model, apply quantization if needed, call `enable_model_cpu_offload()`, then `self._optimize_pipe(self._pipe, compile=getattr(self, "_compile", True))`.
5. In `generate()`: wrap the pipeline call in `torch.inference_mode()`.
6. Add the import and key to `_get_registry()` in `engine.py`.
7. Add a row to the engine table in `CLAUDE.md`.

## Quantization Patterns

- **fp8** (FLUX.1-dev, FLUX.1-schnell): `optimum.quanto.quantize(pipe.transformer, weights=qfloat8)`
- **int8** (SD3.5 T5 encoder): `optimum.quanto.quantize(pipe.text_encoder_3, weights=qint8)`
- **None** (FLUX.2-klein): runs in native bf16, fits in 13 GB without quantization

Always call `freeze(component)` after `quantize()` and before `enable_model_cpu_offload()`.
Wrap quantization in try/except and fall back to bf16 with a warning.

## Pipeline Flow

```
CLI command
    |
    v
preflight_check(engine)   <-- abort if VRAM/RAM insufficient
    |
    v
engine.generate(...)
    |
    +-- engine._load()    <-- download weights, quantize, offload, compile
    |
    +-- torch.inference_mode()
    |
    +-- pipe(prompt, ...)
    |
    +-- image.save(output_path)
    |
    v
[finally] engine.cleanup()
```

## CPU Offload and VRAM Budget

`enable_model_cpu_offload` streams individual model layers to the GPU one at a time,
then moves them back to CPU. This trades throughput for memory: a 13 GB model can run
on a 16 GB card while other processes hold several GB. The 0.85 memory fraction cap
prevents the CUDA allocator from reserving the full device, preserving headroom for
the CUDA runtime and OS.
