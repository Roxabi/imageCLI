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
  │     ├── _get_compute_capability() → (major, minor)
  │     ├── Pipeline.from_pretrained(model_id, torch_dtype=bfloat16)
  │     ├── quantize(transformer, qfloat8 if sm≥(8,9) else qint8)  [FLUX engines]
  │     ├── freeze(transformer)
  │     ├── pipe.enable_model_cpu_offload(gpu_id=0)
  │     ├── torch.cuda.set_per_process_memory_fraction(0.85)
  │     └── _optimize_pipe(pipe)
  ├── build Generator from seed (optional)
  ├── torch.cuda.reset_peak_memory_stats()
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

**Module-level helpers:**

- `_get_compute_capability()` — returns `(major, minor)` for GPU 0, or `(0, 0)` if CUDA unavailable. Used by FLUX engines to select quantization precision at load time.

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
    "pulid-flux2-klein": PuLIDFlux2KleinEngine,
    "flux1-dev": Flux1DevEngine,
    "pulid-flux1-dev": PuLIDFlux1DevEngine,
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

Quantization type is selected at load time based on GPU compute capability:

| GPU arch | sm | FLUX transformer | SD3.5 T5 |
|---|---|---|---|
| Blackwell / Ada Lovelace | ≥ (8,9) | fp8 (`qfloat8`) | int8 |
| Ampere (RTX 3080/3090) | (8,x) | int8 (`qint8`) | int8 |
| No CUDA / older | (0,0) | bf16 fallback | bf16 fallback |

- **fp8** requires sm≥89 (Ada Lovelace, Blackwell) — native hardware support
- **int8** works on Ampere (sm_80+) and newer — halves VRAM vs bf16
- **FLUX.2-klein**: fp8 (`qfloat8`) + CPU offload, peak ~8 GB
- **SD3.5 T5 encoder**: always int8 regardless of GPU

Always call `freeze(component)` after `quantize()` and before `enable_model_cpu_offload()`.
Wrap quantization in try/except and fall back to bf16 with a warning.

## PuLID Architecture

### pulid-flux2-klein

`pulid_flux2_klein.py` injects face identity into the FLUX.2-klein-4B transformer at inference
time via cross-attention (CA) modules. Key design points:

**CA weight loading (`_PuLIDFlux2.from_safetensors`)**

The safetensors file uses the key prefixes `pulid_ca_double.N.*` and `pulid_ca_single.N.*`, which
do not match the `nn.ModuleList` attribute names `double_ca` and `single_ca`. The loader remaps
these prefixes before calling `load_state_dict`. Without this remap, `strict=False` silently
leaves all CA layers randomly initialised — only the IDFormer weights load correctly.

**Auto-detected module counts**

Rather than hardcoding the number of CA modules, `from_safetensors` scans the weight keys and
computes:

```python
n_double = max(int(k.split(".")[1]) for k in state if k.startswith("pulid_ca_double.")) + 1
n_single = max(int(k.split(".")[1]) for k in state if k.startswith("pulid_ca_single.")) + 1
```

The actual Klein v2 weights contain **5 double CA** and **7 single CA** modules (not 12/60).

**Dim mismatch and projection (Klein 4B only)**

| Component | Dim |
|---|---|
| PuLID Klein v2 weights (trained for Klein 9B) | 4096 |
| Klein 4B `hidden_size` | 3072 |

When these differ, `_make_projections(pulid_dim, model_dim, device, dtype)` creates two
orthogonal-init linear layers:

```
proj_up   : Linear(3072 → 4096, bias=False)
proj_down : Linear(4096 → 3072, bias=False)
```

`_apply_ca(ca, hidden_states, id_tokens, projections)` wraps each CA call:

```
hidden_states (3072) → proj_up → trained CA (4096) → proj_down → (3072)
```

The projection layers are randomly initialised, but the trained CA attention patterns that
encode face identity are preserved. This contrasts with the iFayens ComfyUI approach, which
discards all trained CA and initialises random modules at 3072.

**Patching flow**

`_patch_flux(transformer, pulid, id_tokens, strength)` monkey-patches each transformer block's
`forward` method before each generation call and restores the originals in a `finally` block
via the returned `unpatch()` callable. This is why `torch.compile` is disabled: compilation
captures the original (unpatched) forward closures.

### pulid-flux1-dev

Uses PuLID v0.9.1 with `dim=3072` — no projection needed because the model dim matches the
weight dim exactly. Patches `FluxTransformerBlock` and `FluxSingleTransformerBlock` forward
methods (20 CA modules total).

## GPU Support Matrix

| Engine | RTX 5070 Ti (16GB, sm_120) | RTX 3080 (10GB, sm_86) |
|---|---|---|
| `flux2-klein` | ✓ fp8 + cpu_offload | ✗ too large (~13GB) |
| `pulid-flux2-klein` | ✓ bf16 + cpu_offload | ✗ too large |
| `pulid-flux1-dev` | ✓ GGUF Q5_K_S + PuLID | ✗ too large |
| `flux1-dev` | ✓ fp8 | ✓ int8 |
| `flux1-schnell` | ✓ fp8 | ✓ int8 |
| `sd35` | ✓ int8 T5 | ✗ too large (~14GB) |

## VRAM Measurement

`generate()` calls `torch.cuda.reset_peak_memory_stats()` before inference so that
`torch.cuda.max_memory_allocated()` reflects only the inference pass, not model loading.
The CLI reads this value after each generation and prints `Peak VRAM (inference): X.XX GB`.

Use `imagecli info` to see total GPU VRAM and compute capability (`sm_XX (Arch)`).

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
