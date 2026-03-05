# Backend Patterns

Standards for authoring and reviewing engine code in `src/imagecli/`.

## Lazy Engine Loading

Model weights are never loaded on import. Loading happens inside `_load()`, which is
called at the top of `generate()`. Guard with an early return so repeated calls are free:

```python
def _load(self):
    if self._pipe is not None:
        return
    # ... download + setup
```

`__init__` must only set `self._pipe = None`. No torch, no diffusers, no disk I/O.

## Deferred Heavy Imports

`torch`, `diffusers`, and `optimum.quanto` are imported inside `_load()` and `generate()`,
never at module level. This keeps `imagecli engines` and `imagecli info` instant even
without a GPU available.

## Engine Registry

All engines are registered in `engine.py:_get_registry()`. Adding a new engine requires:

1. Create `src/imagecli/engines/<slug>.py` with a class inheriting `ImageEngine`.
2. Add the class import and a kebab-case key to `_get_registry()`.
3. Set `name`, `description`, `model_id`, `vram_gb` as class attributes.

## CPU Offload

Every engine calls `enable_model_cpu_offload(gpu_id=0)` immediately after pipeline
creation. This allows layers to reside in system RAM and stream to GPU on demand,
preventing OOM on 16 GB cards:

```python
self._pipe.enable_model_cpu_offload(gpu_id=0)
torch.cuda.set_per_process_memory_fraction(0.85)
```

`set_per_process_memory_fraction(0.85)` leaves headroom for the CUDA runtime.

## Quantization

- **fp8 (FLUX.1 transformers):** use `optimum.quanto` — `quantize(transformer, weights=qfloat8)` then `freeze(transformer)`. Wrap in try/except and fall back to bf16 with a warning.
- **int8 (text encoders e.g. SD3.5 T5):** same pattern with `qint8`. Text encoders are less sensitive to quantization error than diffusion transformers.

## Performance Optimizations (`_optimize_pipe`)

Call `self._optimize_pipe(self._pipe, compile=...)` at the end of `_load()`. It applies:

- `torch.set_float32_matmul_precision("high")` — uses tensor cores for TF32 matmul (~10-15% speedup on Blackwell).
- `pipe.vae.enable_tiling()` + `pipe.vae.enable_slicing()` — reduces peak VRAM for large resolutions.
- `torch.compile(transformer, mode="max-autotune-no-cudagraphs")` — kernel fusion, 20-40% speedup after first-run warmup. Compiled once per engine instance (`_compiled` flag).

## Memory Safety

`preflight_check(engine)` in `engine.py` must be called before `_load()` in the CLI
layer. It checks free VRAM against `engine.vram_gb` and reads `/proc/meminfo` for
available system RAM (threshold: `IMAGECLI_MIN_FREE_RAM_GB`, default 4.0 GB).

`cleanup()` on the base class calls `torch.cuda.empty_cache()` and `gc.collect()`.
The CLI must call it in a `finally` block after each generation.
