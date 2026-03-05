# Code Review Checklist

Use this checklist when reviewing PRs that touch `src/imagecli/`.

## VRAM Safety

- [ ] `preflight_check(engine)` is called before `_load()` in the CLI layer.
- [ ] `engine.vram_gb` is set to a realistic value (measure with `nvidia-smi`, round up).
- [ ] `enable_model_cpu_offload(gpu_id=0)` is called after pipeline creation.
- [ ] `torch.cuda.set_per_process_memory_fraction(0.85)` is called in `_load()`.

## Resource Cleanup

- [ ] `engine.cleanup()` is called in a `finally` block after generation, so it runs even on failure.
- [ ] No persistent references to pipeline objects outside the engine instance that would block GC.

## Import Discipline

- [ ] `torch`, `diffusers`, `optimum` are imported only inside `_load()` or `generate()`.
- [ ] Module-level imports are limited to stdlib and thin imagecli internals.

## Engine Registration

- [ ] New engine class added to `_get_registry()` in `engine.py`.
- [ ] Engine key is kebab-case and matches the `name` class attribute.
- [ ] Engine class filename is `<slug>.py` under `src/imagecli/engines/`.

## Code Quality

- [ ] `uv run ruff check .` passes with no errors.
- [ ] `uv run ruff format .` produces no diff.
- [ ] No hardcoded paths — use `Path` and project-relative resolution via config.
- [ ] No `print()` outside engine loading progress messages (use Rich console in CLI layer).

## Quantization

- [ ] fp8/int8 quantization wrapped in try/except with bf16 fallback and a printed warning.
- [ ] `freeze()` called after `quantize()` before `enable_model_cpu_offload()`.
