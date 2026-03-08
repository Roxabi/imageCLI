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
| Preflight checks | `preflight_check` raises `InsufficientResourcesError` when VRAM or RAM insufficient |
| Output paths | Non-overwrite suffix logic (`_1`, `_2`, ...) |
| Capabilities | `EngineCapabilities` correctly declared; `warn_ignored_params` triggers on mismatches |
| Effective steps | `effective_steps()` returns fixed value for engines with `fixed_steps` |
| Public API | `imagecli.generate()` calls preflight, delegates to engine, calls cleanup in finally |

## GPU Mocking

No actual model loading or inference in CI. Mock `torch.cuda` for all GPU-related tests:

```python
import unittest.mock as mock

@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.mem_get_info", return_value=(2 * 1024**3, 16 * 1024**3))
def test_preflight_raises_on_low_vram(mock_mem, mock_avail):
    engine = FakeEngine()  # vram_gb = 8.0
    with pytest.raises(InsufficientResourcesError):
        preflight_check(engine)
```

### Mock Boundaries

Mock at the **infrastructure boundary**, not deep inside implementation:

| Test target | Mock at | Don't mock |
|---|---|---|
| `preflight_check()` | `torch.cuda.is_available`, `torch.cuda.mem_get_info` | Engine internals |
| `generate()` facade | `get_engine()`, `preflight_check()` | Config resolution |
| Config loading | File system (`tmp_path`) | `tomllib` itself |
| Markdown parsing | File contents (`tmp_path`) | `yaml.safe_load` |
| Engine registry | Nothing — test the real registry | Don't mock `_get_registry()` |
| Output resolution | File system (`tmp_path`) | `Path` operations |

**Principle:** Mock external dependencies (GPU, filesystem, network), not your own code.
If you find yourself mocking an internal function, the test is probably testing the wrong
thing or the code needs refactoring.

## Test Structure

Follow **Arrange-Act-Assert** (AAA):

```python
def test_resolve_output_appends_suffix_when_exists(tmp_path):
    # Arrange
    (tmp_path / "image.png").touch()
    cfg = {"output_dir": str(tmp_path)}

    # Act
    result = resolve_output(cfg, "image", "png", None)

    # Assert
    assert result == tmp_path / "image_1.png"
```

## Forbidden in CI Tests

- Importing `diffusers`, `optimum`, or real engine classes that trigger `_load()`.
- Writing to `images/images_out/` — use `tmp_path` (pytest fixture) for any file output.
- Sleeping or polling for GPU readiness.
- Network calls to HuggingFace or model downloads.
- Any test that requires a CUDA GPU to pass.

## Fixtures

Use `pytest` built-ins (`tmp_path`, `monkeypatch`) over custom fixtures where possible.
Engine unit tests should instantiate a minimal stub, not a real engine subclass:

```python
class FakeEngine(ImageEngine):
    name = "fake"
    description = "Test stub"
    model_id = "test/model"
    vram_gb = 8.0

    def _load(self):
        pass  # no-op for testing
```

## Test File Naming

```
tests/
  test_config.py       — config.py tests
  test_engine.py       — engine.py tests (registry, preflight, capabilities)
  test_markdown.py     — markdown.py tests
  test_utils.py        — _utils.py tests
  test_init.py         — __init__.py generate() facade tests
```

## Coverage Priorities

Focus testing effort where bugs are most likely and most costly:

| Priority | Area | Why |
|---|---|---|
| High | `preflight_check()` | Wrong check = OOM crash mid-load |
| High | Config priority resolution | Wrong priority = silent wrong defaults |
| High | Output anti-clobber | Bug = overwritten images (data loss) |
| Medium | Engine registry | Wrong registry = `ValueError` on valid name |
| Medium | Markdown parsing | Bad parse = wrong generation parameters |
| Medium | `generate()` facade | Orchestration errors = cleanup not called |
| Lower | Engine capabilities | Cosmetic — user gets a warning or doesn't |
