# Backend Patterns

Standards for authoring and reviewing code in `src/imagecli/`.

## Architecture Principles

These principles are not aspirational — they describe what the codebase already does. New code
must follow them. See [Architecture Patterns](../architecture/patterns.md) for the full
rationale.

**Dependency rule:** Dependencies flow inward. `cli.py` → `__init__.py` → `engine.py` /
`config.py` / `markdown.py`. Never the reverse.

**Single responsibility:** Each module has one job. Don't add config resolution to `engine.py`
or Rich output to `__init__.py`.

**Dependency inversion:** High-level code depends on `ImageEngine` (abstraction), never on
concrete engine classes. Use `get_engine(name)` — don't import `Flux1DevEngine` in `cli.py`.

**Open/closed:** New engines require one file + one registry entry. No changes to `cli.py`
or existing engines.

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

```python
# CORRECT — import inside method
def _load(self):
    import torch
    from diffusers import FluxPipeline
    ...

# WRONG — import at module level
import torch  # slows every import of this module
```

## Engine Registry

All engines are registered in `engine.py:_get_registry()`. Adding a new engine requires:

1. Create `src/imagecli/engines/<slug>.py` with a class inheriting `ImageEngine`.
2. Add the class import and a kebab-case key to `_get_registry()`.
3. Set `name`, `description`, `model_id`, `vram_gb` as class attributes.
4. Declare `capabilities = EngineCapabilities(...)` for parameter support.

## CPU Offload

Every engine calls `enable_model_cpu_offload(gpu_id=0)` immediately after pipeline
creation. This allows layers to reside in system RAM and stream to GPU on demand,
preventing OOM on 16 GB cards:

```python
self._pipe.enable_model_cpu_offload(gpu_id=0)
torch.cuda.set_per_process_memory_fraction(0.85)
```

`set_per_process_memory_fraction(0.85)` leaves headroom for the CUDA runtime.

Use `self._finalize_load(self._pipe)` to apply CPU offload, VRAM cap, and optimizations
in one call — don't duplicate these steps across engines.

## Quantization

- **fp8 (FLUX.1 transformers):** use `self._quantize_transformer(pipe, sm)` from the base
  class. Selects fp8 on sm >= (8,9), int8 on Ampere. Wraps in try/except with bf16 fallback.
- **int8 (text encoders e.g. SD3.5 T5):** use `optimum.quanto` directly —
  `quantize(encoder, weights=qint8)` then `freeze(encoder)`. Wrap in try/except.
- Always call `freeze()` after `quantize()` and before `enable_model_cpu_offload()`.

**Don't** duplicate the quantization-selection logic in each engine. The base class method
handles compute capability detection and fallback.

## Performance Optimizations (`_optimize_pipe`)

Call `self._finalize_load(self._pipe)` at the end of `_load()`. It calls `_optimize_pipe()`
internally, which applies:

- `torch.set_float32_matmul_precision("high")` — TF32 tensor cores (~10-15% speedup).
- `pipe.vae.enable_tiling()` + `pipe.vae.enable_slicing()` — reduces peak VRAM.
- `torch.compile(transformer, mode="max-autotune-no-cudagraphs")` — kernel fusion, 20-40%
  speedup after warmup. Compiled once per engine instance (`_compiled` flag).

## Memory Safety

`preflight_check(engine)` in `engine.py` must be called before `_load()` in the CLI
layer. It checks free VRAM against `engine.vram_gb` and reads `/proc/meminfo` for
available system RAM (threshold: `IMAGECLI_MIN_FREE_RAM_GB`, default 4.0 GB).

`cleanup()` on the base class calls `torch.cuda.empty_cache()` and `gc.collect()`.
The CLI must call it in a `finally` block after each generation.

## Error Handling

imageCLI uses a layered error approach:

| Error type | Where raised | Who catches |
|---|---|---|
| `InsufficientResourcesError` | `preflight_check()` | CLI or library consumer |
| `ValueError` | `get_engine()` (unknown name), `generate()` (bad format) | CLI or library consumer |
| `RuntimeError` | `_quantize_transformer()` (missing optimum) | Engine `_load()` |
| Quantization failures | `_quantize_transformer()` try/except | Caught internally, falls back to bf16 with warning |
| `FileNotFoundError` | `parse_prompt_file()` | CLI or library consumer |

**Principles:**
- **Fail fast:** `preflight_check()` aborts before any model download starts.
- **Graceful degradation:** Quantization failures fall back to bf16, not crash.
- **Domain exceptions:** `InsufficientResourcesError` is a domain concept, not an HTTP error.
  It subclasses `RuntimeError`, not a framework exception.
- **No bare `except:`** — always catch specific exceptions.

## Layer Discipline

| Layer | Allowed | Forbidden |
|---|---|---|
| `cli.py` | Typer, Rich, progress bars | `torch.cuda.*`, model loading, config logic |
| `__init__.py` | Orchestration, re-exports | Rich, Typer, `typer.Exit`, progress bars |
| `engine.py` | ABC, registry, GPU helpers | Rich, Typer, output path logic |
| `engines/*.py` | torch, diffusers (deferred) | Rich, Typer, config reading, output paths |
| `config.py` | stdlib tomllib, Path | torch, engine logic, CLI formatting |
| `markdown.py` | stdlib re, optional yaml | torch, engine logic, CLI formatting |
| `_utils.py` | stdlib Path | torch, engine logic, CLI formatting |

## Template Method Pattern

The base class defines the generation algorithm:

```
generate() → _load() → _build_pipe_kwargs() → pipe() → save()
```

Subclasses override specific steps:

| Hook | Default | Override example |
|---|---|---|
| `_load()` | abstract | Each engine loads its specific pipeline |
| `_build_pipe_kwargs()` | Returns standard kwargs | SD3.5 hardcodes `steps=20`, `guidance=1.0` |
| `effective_steps()` | Returns requested value | SD3.5 returns `20`, Schnell returns `4` |
| `generate()` | Full workflow | Schnell overrides to set default `steps=4` |

## Anti-Patterns

| Don't | Do instead |
|---|---|
| Import concrete engines in `cli.py` | Use `get_engine(name)` |
| Import `torch` at module level | Import inside method body |
| Duplicate quantization logic per engine | Use `self._quantize_transformer()` |
| Call `preflight_check` inside engine | Call it from CLI/facade before `generate()` |
| Print from engine code | Use `logging.getLogger(__name__)` |
| Put output path logic in engines | Receive `output_path` as parameter |
| Add Rich/Typer imports to domain modules | Keep framework deps in `cli.py` only |
| Catch generic `Exception` in engines | Catch specific exceptions; let unexpected errors propagate |

## AI Quick Reference

When writing or reviewing engine code, verify:

1. `_load()` starts with `if self._pipe is not None: return`
2. `torch` / `diffusers` imported inside methods, not at top
3. `_finalize_load(self._pipe)` called at end of `_load()`
4. `EngineCapabilities` declared with correct `fixed_steps` / `fixed_guidance` / `negative_prompt`
5. Engine registered in `_get_registry()` with kebab-case key matching `name`
6. No imports from `cli.py` or other engines
7. Quantization uses base class `_quantize_transformer()` for FLUX engines
