# Code Review Checklist

Use this checklist when reviewing PRs that touch `src/imagecli/`.

## Architecture

- [ ] Dependencies flow inward — no imports from outer layers into inner layers.
- [ ] `cli.py` does not import concrete engine classes — uses `get_engine(name)`.
- [ ] No `torch`, `diffusers`, `rich`, or `typer` imports in modules where they don't belong
      (see [layer discipline table](./backend-patterns.md#layer-discipline)).
- [ ] New public functions exposed in `__init__.py` if they're part of the library API.
- [ ] No circular imports introduced.

## VRAM Safety

- [ ] `preflight_check(engine)` is called before `_load()` in the CLI layer.
- [ ] `engine.vram_gb` is set to a realistic value (measure with `nvidia-smi`, round up).
- [ ] `enable_model_cpu_offload(gpu_id=0)` is called after pipeline creation, or 2-phase batch methods are implemented when `supports_two_phase = True`.
- [ ] `torch.cuda.set_per_process_memory_fraction(0.85)` is called in `_load()`.
- [ ] New engine uses `self._finalize_load(self._pipe)` instead of manual setup.

## Resource Cleanup

- [ ] `engine.cleanup()` is called in a `finally` block after generation, so it runs even
      on failure.
- [ ] No persistent references to pipeline objects outside the engine instance that would
      block GC.
- [ ] Batch mode cleans up all cached engines after the full batch completes.

## Import Discipline

- [ ] `torch`, `diffusers`, `optimum` are imported only inside `_load()` or `generate()`.
- [ ] Module-level imports are limited to stdlib and thin imagecli internals.
- [ ] No new heavy imports at module level that would slow `imagecli engines` / `imagecli info`.

## Engine Registration

- [ ] New engine class added to `_get_registry()` in `engine.py`.
- [ ] Engine key is kebab-case and matches the `name` class attribute.
- [ ] Engine class filename is `<slug>.py` under `src/imagecli/engines/`.
- [ ] `EngineCapabilities` declared with correct `fixed_steps`, `fixed_guidance`,
      `negative_prompt`.

## Template Method Compliance

- [ ] `_load()` starts with `if self._pipe is not None: return` (idempotent guard).
- [ ] `_finalize_load(self._pipe)` called at the end of `_load()`.
- [ ] `_build_pipe_kwargs()` overridden if the engine has fixed/ignored parameters.
- [ ] `effective_steps()` overridden if the engine has a fixed step count.

## Error Handling

- [ ] Domain errors use `InsufficientResourcesError` or `ValueError`, not framework
      exceptions.
- [ ] Quantization wrapped in try/except with bf16 fallback and logged warning.
- [ ] `freeze()` called after `quantize()` before `enable_model_cpu_offload()` (or before the generation phase for 2-phase engines).
- [ ] No bare `except:` — always catch specific exceptions.
- [ ] Errors propagate to the caller — engines don't swallow exceptions silently.

## Code Quality

- [ ] `uv run ruff check .` passes with no errors.
- [ ] `uv run ruff format .` produces no diff.
- [ ] No hardcoded paths — use `Path` and project-relative resolution via config.
- [ ] No `print()` — use `logging.getLogger(__name__)` in engines, Rich console in CLI.
- [ ] No unnecessary abstractions or helpers for one-time operations.

## Testing

- [ ] New logic has corresponding tests in `tests/`.
- [ ] Tests mock at infrastructure boundaries, not internal functions.
- [ ] Tests use `tmp_path` for file output, not `images/images_out/`.
- [ ] No tests require a GPU or network access.
- [ ] `uv run pytest` passes.
