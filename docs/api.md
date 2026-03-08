# Python API Reference

imagecli can be used as a Python library in addition to the CLI. All public symbols are importable directly from the `imagecli` package.

```python
from imagecli import generate, get_engine, list_engines
```

## Quick examples

**One-liner generation:**

```python
from imagecli import generate

path = generate("a white cat on a red chair")
print(path)  # PosixPath('images/images_out/image.png')
```

**With options:**

```python
from imagecli import generate

path = generate(
    "a white cat on a red chair",
    engine="flux1-dev",
    width=1280,
    height=720,
    steps=30,
    guidance=3.5,
    seed=42,
    negative_prompt="blurry, low quality",
    format="webp",
)
```

**Engine reuse across multiple prompts (avoids reloading the model):**

```python
from imagecli import get_engine, preflight_check
from pathlib import Path

eng = get_engine("flux1-schnell", compile=False)
preflight_check(eng)

prompts = ["a cat", "a dog", "a fox"]
try:
    for i, prompt in enumerate(prompts):
        eng.generate(prompt, output_path=Path(f"out/image_{i}.png"))
finally:
    eng.cleanup()
```

**Using a `.md` prompt file:**

```python
from imagecli import generate, parse_prompt_file
from pathlib import Path

doc = parse_prompt_file(Path("images/prompts_in/my_scene.md"))
path = generate(
    doc.prompt,
    engine=doc.engine,
    width=doc.width,
    height=doc.height,
    steps=doc.steps,
    guidance=doc.guidance,
    seed=doc.seed,
    negative_prompt=doc.negative_prompt,
    format=doc.format,
)
```

---

## Functions

### `generate`

```python
def generate(
    prompt: str,
    *,
    engine: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    negative_prompt: str = "",
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    format: str | None = None,
    compile: bool = True,
) -> Path
```

Generate an image from a text prompt. Returns the `Path` to the saved file.

`None` parameters fall back to values from `imagecli.toml`, then to hardcoded defaults. Priority: **kwarg > `imagecli.toml` > hardcoded default**.

Hardcoded defaults when no config file is found: `engine="flux2-klein"`, `width=1024`, `height=1024`, `steps=50`, `guidance=4.0`, `format="png"`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The text prompt to generate from. |
| `engine` | `str \| None` | Engine name. One of `flux2-klein`, `flux1-dev`, `flux1-schnell`, `sd35`. |
| `width` | `int \| None` | Image width in pixels. Must be a multiple of 64. |
| `height` | `int \| None` | Image height in pixels. Must be a multiple of 64. |
| `steps` | `int \| None` | Number of inference steps. Ignored by `sd35` (fixed at 20) and `flux1-schnell` (fixed at 4). |
| `guidance` | `float \| None` | CFG guidance scale. Ignored by `sd35` (fixed at 1.0) and `flux1-schnell` (fixed at 0.0). |
| `seed` | `int \| None` | Random seed for reproducibility. `None` means non-deterministic. |
| `negative_prompt` | `str` | What to avoid in the image. Not supported by all engines. |
| `output_path` | `str \| Path \| None` | Exact output file path. Parent directories are created automatically. Takes precedence over `output_dir`. |
| `output_dir` | `str \| Path \| None` | Directory to save the image in. Filename is auto-generated as `image.{format}`, with `_1`, `_2`, etc. appended if the file already exists. |
| `format` | `str \| None` | Output format: `png`, `jpg`, or `webp`. |
| `compile` | `bool` | Enable `torch.compile` on the model transformer (default `True`). Set to `False` for faster startup at the cost of slower inference. |

**Behavior:**

- Calls `load_config()` internally to resolve defaults. `UserWarning` is suppressed if no config file is found.
- Calls `preflight_check()` before loading the model. Raises `InsufficientResourcesError` if VRAM or RAM is insufficient.
- Calls `eng.cleanup()` in a `finally` block — VRAM is freed even if generation fails.
- Output files are never overwritten. When `output_path` is not set, the suffix `_1`, `_2`, etc. is appended automatically. When `output_path` is set explicitly, no anti-clobber is applied — the caller controls the exact path.
- **Security note:** `output_path` and `output_dir` are used as-is without path validation. If your application accepts paths from untrusted input, validate them before passing to `generate()`.

**Raises:**

- `InsufficientResourcesError` — not enough VRAM or system RAM.
- `ValueError` — unknown engine name.

---

### `get_engine`

```python
def get_engine(name: str, *, compile: bool = True) -> ImageEngine
```

Return a new, unloaded `ImageEngine` instance for the given engine name. The model is not loaded until the first call to `eng.generate()`.

Use this for manual control over the engine lifecycle — for example, to reuse a loaded model across multiple generations without the overhead of reloading.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Engine name. One of `flux2-klein`, `flux1-dev`, `flux1-schnell`, `sd35`. |
| `compile` | `bool` | Enable `torch.compile` when the engine loads (default `True`). |

**Raises:**

- `ValueError` — unknown engine name, with the list of valid names included in the message.

---

### `list_engines`

```python
def list_engines() -> list[dict]
```

Return a list of all registered engines. Each entry is a dict with the following keys:

| Key | Type | Description |
|---|---|---|
| `name` | `str` | Engine identifier used in CLI and `get_engine`. |
| `description` | `str` | Short human-readable description. |
| `model_id` | `str` | HuggingFace model ID. |
| `vram_gb` | `float` | Approximate minimum VRAM required in GB. |

Does not require a GPU and makes no network calls.

---

### `preflight_check`

```python
def preflight_check(engine: ImageEngine) -> None
```

Check that the system has enough resources to load the engine. Raises `InsufficientResourcesError` immediately if the check fails, before any model download or load begins.

Skipped silently when the engine is already loaded (i.e. `_pipe` is not `None`).

**Checks performed:**

1. **CUDA GPU** — raises if no CUDA-capable GPU is detected.
2. **Free VRAM** — raises if free VRAM is below `engine.vram_gb`.
3. **Free system RAM** — raises if `MemAvailable` from `/proc/meminfo` is below `IMAGECLI_MIN_FREE_RAM_GB` (default `4.0` GB). This check is skipped on non-Linux systems.

**Environment variables:**

- `IMAGECLI_MIN_FREE_RAM_GB` — minimum free system RAM in GB (default `4.0`).

**Raises:**

- `InsufficientResourcesError` — VRAM or RAM is insufficient, or no CUDA GPU is present.

---

### `parse_prompt_file`

```python
def parse_prompt_file(path: Path) -> PromptDoc
```

Parse a `.md` prompt file with optional YAML frontmatter. Returns a `PromptDoc`.

The file format is:

```markdown
---
engine: flux1-dev
width: 1280
height: 720
steps: 30
guidance: 3.5
seed: 42
negative_prompt: "blurry"
format: png
---

Your prompt text here. Can be multiple paragraphs.
```

All frontmatter fields are optional. If there is no frontmatter, the entire file content becomes the prompt. If the frontmatter contains a `prompt` key, that value is used as the prompt and the body is ignored.

**Raises:**

- `FileNotFoundError` — path does not exist.

---

### `load_config`

```python
def load_config() -> dict
```

Load `imagecli.toml` by walking up from the current working directory to `$HOME`. Returns the merged configuration dict.

Hardcoded defaults are always present in the returned dict. Values from `imagecli.toml` under `[defaults]` override them.

**Returned keys:**

| Key | Type | Default |
|---|---|---|
| `engine` | `str` | `"flux2-klein"` |
| `width` | `int` | `1024` |
| `height` | `int` | `1024` |
| `steps` | `int` | `50` |
| `guidance` | `float` | `4.0` |
| `output_dir` | `str` | `"images/images_out"` |
| `format` | `str` | `"png"` |
| `quality` | `int` | `95` |
| `_config_path` | `str` | Present only when a config file was found. Absolute path to the loaded file. |

When no config file is found, a `UserWarning` is emitted and defaults are returned.

---

## Classes

### `ImageEngine`

```python
class ImageEngine(ABC)
```

Abstract base class for all image generation engines. Use this for type hints when writing code that accepts any engine.

**Class attributes** (must be set on subclasses):

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Engine identifier (e.g. `"flux2-klein"`). |
| `description` | `str` | Short description shown by `imagecli engines`. |
| `model_id` | `str` | HuggingFace model ID. |
| `vram_gb` | `float` | Approximate minimum VRAM required in GB. |

**Key methods:**

`generate(prompt, *, output_path, negative_prompt="", width=1024, height=1024, steps=50, guidance=4.0, seed=None, **kwargs) -> Path`
Run inference and save the result to `output_path`. Calls `_load()` on the first invocation.

`cleanup() -> None`
Free GPU memory. Calls `torch.cuda.empty_cache()` and `gc.collect()`. Always call this in a `finally` block when managing an engine manually.

`effective_steps(requested: int) -> int`
Return the number of steps that will actually run. Engines with a fixed step count (e.g. SD3.5 Turbo, FLUX.1-schnell) override this to return their hardcoded value.

---

### `PromptDoc`

```python
@dataclass
class PromptDoc:
    prompt: str
    negative_prompt: str = ""
    engine: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    format: str | None = None
    extra: dict = field(default_factory=dict)
```

Dataclass returned by `parse_prompt_file`. All fields except `prompt` are optional and will be `None` when not specified in the frontmatter.

`extra` contains any frontmatter keys not recognised by imagecli (useful for custom metadata).

---

### `InsufficientResourcesError`

```python
class InsufficientResourcesError(RuntimeError)
```

Raised by `preflight_check` when the system cannot safely load an engine. Subclasses `RuntimeError`.

Catch this to handle resource failures gracefully:

```python
from imagecli import generate, InsufficientResourcesError

try:
    path = generate("a cat", engine="flux1-dev")
except InsufficientResourcesError as e:
    print(f"Not enough resources: {e}")
```

The error message includes the specific threshold that was not met and a suggestion (e.g. closing other GPU processes).

---

## See also

- CLI reference: see the `imagecli` commands in [CLAUDE.md](../CLAUDE.md).
- Config file format and all settable keys: `imagecli.example.toml` in the project root.
- Adding a new engine: [CONTRIBUTING.md](../CONTRIBUTING.md).
