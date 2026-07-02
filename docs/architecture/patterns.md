# Architecture Patterns — Clean Architecture + Hexagonal

## Overview

imageCLI follows **clean architecture** and **hexagonal (ports and adapters)** patterns adapted
for a Python CLI tool. These patterns were not adopted speculatively — they emerged from real
constraints: supporting multiple GPU backends that must be swappable, keeping CLI startup fast
despite multi-GB model dependencies, and enabling both CLI and library usage from the same
codebase.

This document describes the patterns **already in the code**, not aspirational targets. If the
code and this document disagree, the code is authoritative — update this document.

## When These Patterns Matter

Not every module needs hexagonal architecture. Use this table to decide whether a change
warrants applying these patterns or whether direct, simple code is sufficient.

| Observable signal | Pattern to apply |
|---|---|
| New image generation backend needs to be added | Implement the `ImageEngine` port (abstract base) |
| Heavy library (`torch`, `diffusers`) would slow down CLI startup | Deferred import (import inside method, not at module level) |
| Two modules need the same hardware-detection logic | Centralise in `engine.py` (see [ADR-001](./adr/001-hardware-capability-detection-boundary.mdx)) |
| CLI display logic is mixing with domain logic | Move domain logic to `engine.py` or `config.py`; keep `cli.py` as pure presentation |
| A function needs to work from both CLI and library API | Put it in the domain layer, expose via `__init__.py` |
| GPU resource check is needed before expensive operation | Use `preflight_check()` — fail fast, not fail mid-load |

If none of these signals are present, write simple, direct code. Don't add abstractions for
hypothetical future needs.

## Layer Model

Three concentric layers. The **dependency rule** is strict: outer layers depend on inner
layers, never the reverse.

```
┌─────────────────────────────────────────────────┐
│           Presentation (outermost)              │
│  cli.py — Typer, Rich, progress bars            │
│  Translates user input → domain calls           │
│  Formats domain output → Rich console           │
└────────────────────┬────────────────────────────┘
                     │ depends on
                     ▼
┌─────────────────────────────────────────────────┐
│           Domain / Application                  │
│  __init__.py  — generate() facade               │
│  engine.py    — ImageEngine ABC, registry       │
│  config.py    — config resolution               │
│  markdown.py  — prompt file parsing             │
│  _utils.py    — output path resolution          │
│  (No framework deps — pure Python)              │
└────────────────────┬────────────────────────────┘
                     │ depends on (deferred)
                     ▼
┌─────────────────────────────────────────────────┐
│           Infrastructure (innermost*)           │
│  engines/*.py — concrete pipeline adapters      │
│  torch, diffusers, optimum[quanto]              │
│  (Imported at runtime, inside _load() only)     │
└─────────────────────────────────────────────────┘
```

*\* In classic hexagonal architecture, infrastructure is the outermost ring. Here, engine
adapters live "below" the domain because they implement the abstract port defined in the
domain layer. The dependency still points inward: `engines/flux1_dev.py` imports from
`engine.py`, never the reverse.*

## Hexagonal Patterns in imageCLI

### Port: `ImageEngine` (Abstract Base Class)

The **port** is `ImageEngine` in `engine.py`. It defines the contract that all backends must
satisfy:

```python
# engine.py — domain layer
class ImageEngine(ABC):
    name: str               # registry key
    description: str        # human-readable label
    model_id: str           # HuggingFace repo ID
    vram_gb: float          # minimum VRAM requirement
    capabilities: ClassVar[EngineCapabilities]

    @abstractmethod
    def _load(self) -> None: ...

    def generate(self, prompt, *, output_path, ...) -> Path: ...
    def cleanup(self) -> None: ...
    def effective_steps(self, requested: int) -> int: ...
    def _build_pipe_kwargs(self, ...) -> dict: ...
```

The port also provides **shared infrastructure** via concrete helper methods:
- `_finalize_load(pipe)` — CPU offload, VRAM cap, optimizations (standard path; 2-phase engines use separate load/encode/generate methods instead)
- `_optimize_pipe(pipe, *, compile: bool | None = None)` — TF32, VAE tiling/slicing, `torch.compile`
- `_quantize_transformer(pipe, sm)` — adaptive fp8/int8 quantization
- `_save_image(image, output_path)` — shared image persistence helper
- `load_for_encode()`, `encode_prompt()`, `start_generation_phase()`, `generate_from_embeddings()` — 2-phase batch hooks (override when `supports_two_phase = True`)
- `cleanup()` — VRAM cache clearing + garbage collection

This is the **Template Method** pattern: the base class defines the algorithm skeleton
(`generate` → `_load` → `_build_pipe_kwargs` → inference → save), and subclasses override
specific steps.

### Adapters: Concrete Engines

Each engine in `engines/` is an **adapter** that implements the port using a specific
Diffusers pipeline:

```python
# engines/flux1_dev.py — infrastructure layer
class Flux1DevEngine(ImageEngine):
    name = "flux1-dev"
    model_id = "black-forest-labs/FLUX.1-dev"
    vram_gb = 10.0
    capabilities = EngineCapabilities(negative_prompt=False)

    def _load(self):
        if self._pipe is not None:
            return
        from diffusers import FluxPipeline
        self._pipe = FluxPipeline.from_pretrained(self.model_id, ...)
        self._quantize_transformer(self._pipe, sm)
        self._finalize_load(self._pipe)
```

**Key adapter properties:**
- Adapters import `torch`/`diffusers` only inside `_load()` — never at module level
- Adapters depend on the port (`ImageEngine`), never on each other
- Adapters are registered in `_get_registry()` — adding a new backend means adding one file
  and one registry entry
- Adapters are **stateful** — they hold the loaded pipeline and are reused across generations

### Registry: Strategy Pattern

`engine.py:_get_registry()` maps engine names to classes. `get_engine(name)` instantiates
the selected adapter. This is the **Strategy pattern**: the caller selects a strategy (engine)
by name, and the system provides the correct implementation.

```python
def _get_registry() -> dict[str, type[ImageEngine]]:
    return {
        "flux2-klein": Flux2KleinEngine,
        "flux1-dev": Flux1DevEngine,
        "flux1-schnell": Flux1SchnellEngine,
        "sd35": SD35Engine,
    }
```

The registry uses **deferred imports** — engine classes are imported inside the function body,
not at module level. This keeps `imagecli engines` and `imagecli info` fast even without a
GPU.

### Facade: `__init__.py`

The `generate()` function in `__init__.py` is a **Facade** that hides the multi-step
orchestration (config resolution → engine instantiation → preflight → generation → cleanup)
behind a single function call:

```python
from imagecli import generate
path = generate("a cat in space")  # one call does everything
```

Library consumers never need to touch `engine.py`, `config.py`, or `_utils.py` directly.

## Clean Architecture Principles Applied

### Single Responsibility

| Module | One job |
|---|---|
| `cli.py` | Translate CLI input/output (Typer ↔ domain) |
| `__init__.py` | Orchestrate the generate workflow |
| `engine.py` | Define the engine contract + shared GPU logic |
| `engines/*.py` | Load and run one specific model |
| `config.py` | Find and parse `imagecli.toml` |
| `markdown.py` | Parse `.md` prompt files |
| `_utils.py` | Resolve output file paths |

### Dependency Inversion

High-level modules (`cli.py`, `__init__.py`) depend on the abstraction (`ImageEngine`), not
on concrete engines. `cli.py` never imports `Flux1DevEngine` — it calls `get_engine("flux1-dev")`
and receives an `ImageEngine` instance.

### Open/Closed

Adding a new engine requires:
1. Create `engines/new_engine.py` implementing `ImageEngine`
2. Add one entry to `_get_registry()`
3. Done — no changes to `cli.py`, `__init__.py`, or any other engine

The system is **open for extension** (new engines) and **closed for modification** (existing
code doesn't change).

### Interface Segregation

`EngineCapabilities` declares what each engine supports. The CLI reads capabilities to warn
users about ignored parameters — it never needs to know engine internals. Engines declare
their limits; the CLI respects them.

## Infrastructure Boundary: Deferred Imports

The most important architectural pattern in imageCLI is the **deferred import boundary**.
Heavy libraries (`torch`, `diffusers`, `optimum.quanto`) are never imported at module level.
They are imported inside method bodies (`_load()`, `generate()`, `preflight_check()`).

This creates a hard boundary between the domain layer (fast, no GPU required) and the
infrastructure layer (slow to import, requires CUDA):

```python
# DOMAIN — instant, no GPU
from imagecli import list_engines, load_config, parse_prompt_file

# INFRASTRUCTURE — triggers torch import only when called
engine = get_engine("flux1-dev")
engine.generate(...)  # torch/diffusers imported here
```

**Why this matters:**
- `imagecli engines` lists engines without loading PyTorch (~5s import)
- `imagecli info` shows config without a GPU
- Library consumers can inspect engines without triggering model downloads
- Tests run without `torch` installed (mock at the boundary)

## What Does NOT Belong in Each Layer

| Layer | Keep out | Why |
|---|---|---|
| `cli.py` (presentation) | `torch.cuda.*` calls, model loading, config resolution logic | CLI is display only; domain owns logic |
| `engine.py` (domain) | `typer`, `rich`, `Console`, progress bars | Domain must work without a CLI |
| `engines/*.py` (infrastructure) | Output path logic, config reading, CLI formatting | Adapters do one thing: load and run a model |
| `config.py` (domain) | Engine selection logic, default parameter merging for CLI flags | Config reads TOML; priority resolution belongs in the orchestrator |
| `__init__.py` (facade) | Rich output, progress bars, `typer.Exit` | Facade is for library consumers — no CLI deps |

## Anti-Patterns to Avoid

| Anti-pattern | Example | Correct approach |
|---|---|---|
| Hardware detection in CLI | `cli.py` building a `{major: "Blackwell"}` dict | Centralise in `engine.py:get_gpu_info()` ([ADR-001](./adr/001-hardware-capability-detection-boundary.mdx)) |
| Importing torch at module level | `import torch` at top of `engine.py` | Import inside method body: `def _load(self): import torch` |
| Engine knowing about output paths | Engine deciding where to save files | Engine receives `output_path` as parameter; caller resolves it |
| Duplicating quantization logic | Each engine copy-pasting fp8/int8 selection | Use `self._quantize_transformer(pipe, sm)` from base class |
| Concrete engine imports in CLI | `from imagecli.engines.flux1_dev import Flux1DevEngine` in `cli.py` | Use `get_engine("flux1-dev")` — go through the registry |
| Mixing domain logic with display | `engine.generate()` printing to console | Engines use `logging`; CLI converts log output to Rich |

## Related Documentation

- [Architecture Overview](./index.md) — Layer diagram and module map
- [Engine Architecture](./engines.md) — Engine lifecycle, quantization, registry details
- [Ubiquitous Language](./ubiquitous-language.md) — Domain glossary
- [Backend Patterns](../standards/backend-patterns.md) — Coding standards for engine code
