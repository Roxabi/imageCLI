# Testing Standards

## Runner

```
uv run pytest -v
```

All tests live under `tests/`. File names follow `test_<module>.py`.

## What to Test

| Area | What to cover |
|---|---|
| Config loading | TOML discovery (walk-up), priority order (CLI > frontmatter > toml > default), missing file |
| Engine registry | `_get_registry()` returns all expected keys; `get_engine()` raises on unknown name |
| Markdown parsing | Frontmatter extracted correctly; missing keys use defaults; body stripped |
| Memory checks | `preflight_check` raises `MemoryError` when VRAM or RAM is insufficient |
| Output paths | Non-overwrite suffix logic (`_1`, `_2`, …) |

## GPU Mocking

No actual model loading or inference in CI. Mock `torch.cuda` for all GPU-related tests:

```python
import unittest.mock as mock

@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.mem_get_info", return_value=(2 * 1024**3, 16 * 1024**3))
def test_preflight_raises_on_low_vram(mock_mem, mock_avail):
    engine = FakeEngine()  # vram_gb = 8.0
    with pytest.raises(MemoryError):
        preflight_check(engine)
```

## Forbidden in CI Tests

- Importing `diffusers`, `optimum`, or real engine classes that trigger `_load()`.
- Writing to `images/images_out/` — use `tmp_path` (pytest fixture) for any file output.
- Sleeping or polling for GPU readiness.

## Fixtures

Use `pytest` built-ins (`tmp_path`, `monkeypatch`) over custom fixtures where possible.
Engine unit tests should instantiate a minimal stub, not a real engine subclass.
