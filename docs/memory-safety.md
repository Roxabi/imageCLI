# Memory Safety

`preflight_check(engine)` runs before every load and raises `MemoryError` (a `RuntimeError` subclass) if:

- Free VRAM < `engine.vram_gb` (checked via `torch.cuda.mem_get_info`)
- Free system RAM < `IMAGECLI_MIN_FREE_RAM_GB` env var (default `4.0`, reads `/proc/meminfo`)
- No CUDA GPU detected

`cleanup()` is called in a `finally` block after every generation: `torch.cuda.empty_cache()` + `gc.collect()`.

This guarantees:
- Early failure on insufficient VRAM (no mid-load OOM).
- VRAM reclamation even if generation raises.
