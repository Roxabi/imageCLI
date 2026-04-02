# Architecture Overview

## System Map

imageCLI is a Python CLI and library for local GPU-accelerated image generation. It wraps multiple HuggingFace Diffusers backends behind a unified interface.

```
┌─────────────────────────────────────────────────────────────┐
│                 Presentation Layer                          │
│  cli.py — Typer commands, Rich console, progress bars      │
│  (Typer, Rich imported here only)                          │
└────────────────────────┬────────────────────────────────────┘
                         │ imports
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   API / Facade Layer                        │
│  __init__.py — generate(), get_engine(), list_engines()     │
│  Public surface for library consumers                      │
│  (No framework deps — pure Python orchestration)           │
└────────────────────────┬────────────────────────────────────┘
                         │ imports
                         ▼
┌──────────────────────────────┬──────────────────────────────┐
│     Domain Infrastructure    │     Engine Subsystem         │
├──────────────────────────────┼──────────────────────────────┤
│ config.py                    │ engine.py                    │
│   TOML loader (walks CWD→~) │   ImageEngine (abstract)     │
│   Priority resolution        │   Registry + factory         │
│   stdlib only                │   preflight_check()          │
│                              │   EngineCapabilities         │
│ markdown.py                  │   _quantize_transformer()    │
│   YAML frontmatter parser    │   _finalize_load()           │
│   PromptDoc dataclass        │   _optimize_pipe()           │
│   stdlib + optional yaml     │                              │
│                              │ engines/ (adapters)          │
│ _utils.py                    │   flux2_klein.py             │
│   Output path resolution     │   flux1_dev.py               │
│   Anti-clobber suffixing     │   flux1_schnell.py           │
│   stdlib only                │   sd35.py                    │
└──────────────────────────────┴──────────────────────────────┘
                         │ imports (deferred, at runtime)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               External Dependencies                         │
│  torch, diffusers, optimum[quanto]  (GPU inference)        │
│  typer, rich  (CLI only, never in domain)                  │
│  pyyaml  (optional, markdown parsing)                      │
└─────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

| Module | Layer | Responsibility | External deps |
|---|---|---|---|
| `cli.py` | Presentation | CLI commands, Rich output, progress bars | typer, rich |
| `__init__.py` | API Facade | `generate()` function, re-exports public API | none |
| `engine.py` | Domain | Abstract engine, registry, preflight, capabilities | deferred torch |
| `engines/*.py` | Infrastructure | Concrete engine adapters (model loading, inference) | deferred torch, diffusers |
| `config.py` | Domain | TOML config discovery and priority resolution | stdlib tomllib |
| `markdown.py` | Domain | Prompt file parsing, PromptDoc dataclass | optional pyyaml |
| `_utils.py` | Domain | Output path resolution, anti-clobber logic | stdlib only |

## Dependency Flow

Dependencies flow **strictly inward** — outer layers depend on inner layers, never the reverse.

```
cli.py  ──imports──►  __init__.py  ──imports──►  engine.py
                                   ──imports──►  config.py
                                   ──imports──►  markdown.py
                                   ──imports──►  _utils.py

engines/flux1_dev.py  ──imports──►  engine.py (abstract base only)
engines/sd35.py       ──imports──►  engine.py (abstract base only)
```

**Zero upward dependencies:**
- `engine.py` does NOT import from `cli.py`
- `config.py` does NOT import from any imagecli module
- `markdown.py` does NOT import from any imagecli module
- Concrete engines import ONLY from `engine.py` (their abstract base)

**Zero circular dependencies** — the import graph is a DAG.

## Data Flow

### Single generation

```
User ──"a cat in space"──► CLI generate()
                               │
                               ▼
                          load_config()  ──► imagecli.toml (optional)
                               │
                               ▼
                          resolve priority: CLI flag > frontmatter > config > default
                               │
                               ▼
                          get_engine(name)  ──► _get_registry() ──► Flux2KleinEngine()
                               │
                               ▼
                          preflight_check(engine)  ──► VRAM + RAM sufficient?
                               │                      ✗ → InsufficientResourcesError
                               ▼
                          engine.generate(prompt, ...)
                               │
                               ├── _load()  ──► download weights, quantize, CPU offload, compile
                               │               (idempotent — returns early if already loaded)
                               │               (flux2-klein batch uses 2-phase: encode all prompts,
                               │                then generate all images — no CPU offload in that path)
                               │
                               ├── torch.inference_mode()
                               │
                               ├── pipe(prompt, width, height, steps, guidance)
                               │
                               └── image.save(output_path)
                               │
                               ▼
                          [finally] engine.cleanup()  ──► empty_cache + gc.collect
```

### Batch generation

Same as single generation, but:
- Engine instances are **cached** across files (one model load per engine type)
- Cleanup runs **once** at the end, after all files are processed
- Each `.md` file's frontmatter can specify a different engine
- Engines with `supports_two_phase = True` (e.g. `flux2-klein`) use a 2-phase path: all prompts are encoded first (Phase 1), then all images are generated (Phase 2), avoiding per-image CPU offload overhead

## Entry Points

| Entry point | Target audience | What it provides |
|---|---|---|
| `imagecli` CLI (via `cli.py`) | End users | Rich output, progress bars, `--help`, batch mode |
| `from imagecli import generate` | Python developers | Framework-neutral API, no CLI deps |
| `from imagecli import get_engine` | Advanced users | Manual engine lifecycle control |

## Related Documentation

- [Architecture Patterns](./patterns.md) — How the codebase maps to clean/hexagonal architecture
- [Engine Architecture](./engines.md) — Engine lifecycle, quantization, registry, GPU support
- [Ubiquitous Language](./ubiquitous-language.md) — Domain glossary
- [ADRs](./adr/) — Architecture Decision Records
