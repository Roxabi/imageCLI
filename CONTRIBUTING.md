# Contributing

## Project Structure

```
src/imagecli/
  cli.py            — Typer app: generate, batch, engines, info commands
  config.py         — TOML config loader (walks up from CWD to $HOME)
  engine.py         — ImageEngine base class, registry (_get_registry), preflight checks
  markdown.py       — YAML frontmatter parser for .md prompt files
  engines/
    flux2_klein.py        — FLUX.2-klein-4B (default)
    pulid_flux2_klein.py  — FLUX.2-klein-4B + PuLID face lock (optional extra)
    pulid_flux1_dev.py    — FLUX.1-dev GGUF + PuLID v0.9.1 face lock
    flux1_dev.py          — FLUX.1-dev fp8 quantized
    flux1_schnell.py      — FLUX.1-schnell
    sd35.py               — SD3.5 Large Turbo
images/
  prompts_in/       — .md prompt files (tracked in git)
  images_out/       — generated images (gitignored)
```

## Development Setup

```bash
git clone <repo>
cd imageCLI
uv sync
uv run imagecli info
```

Requires a CUDA-capable GPU with at least 10 GB free VRAM. PyTorch is pulled from
the PyTorch cu128 index automatically by `uv sync`.

Copy `imagecli.example.toml` to `~/imagecli.toml` and adjust defaults if needed.

## Conventions

- Line length 100, target Python 3.12 — enforced by `ruff` (see `pyproject.toml`).
- Keep the CLI thin. Logic belongs in engine classes, not in `cli.py`.
- Heavy imports (`torch`, `diffusers`) must be deferred inside `_load()`. Do not
  import them at module level or in `__init__`.
- Engines are lazy-loaded: the model is not instantiated until the first call.
- No over-engineering. Flat structure, simple patterns.
- Output files never overwrite — suffix `_1`, `_2`, etc. is appended automatically.

## Adding a New Engine

1. Create `src/imagecli/engines/foo.py`. Subclass `ImageEngine` from `engine.py`:

   ```python
   from imagecli.engine import ImageEngine

   class FooEngine(ImageEngine):
       name = "foo"
       description = "Short description shown in imagecli engines"
       model_id = "org/model-id"
       vram_gb = 12.0

       def _load(self) -> None:
           # Import torch/diffusers here, not at the top of the file
           from diffusers import SomePipeline
           self._pipe = SomePipeline.from_pretrained(self.model_id)
           self._pipe.enable_model_cpu_offload()
           self._optimize_pipe(self._pipe)
           # Note: engines that set supports_two_phase = True implement
           # load_for_encode(), encode_prompt(), start_generation_phase(),
           # and generate_from_embeddings() instead of using CPU offload
           # for the batch path. See flux2_klein.py for a reference.

       def generate(self, prompt, *, output_path, **kwargs):
           if not hasattr(self, "_pipe"):
               self._load()
           # run inference, save to output_path, return output_path
   ```

2. Register the engine in `engine.py` inside `_get_registry()`:

   ```python
   from imagecli.engines.foo import FooEngine
   return {
       ...
       "foo": FooEngine,
   }
   ```

3. Add a row to the engines table in `CLAUDE.md`.

## Quality Gates

Run before opening a PR:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

All three must pass with no errors. Tests live in `tests/` and must not require a GPU —
mock `torch` and `diffusers` for unit tests.

## Commit Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>
```

Common types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`  
Common scopes: `cli`, `engine`, `markdown`, `config`, `engines/<name>`

Examples:

```
feat(engine): add SDXL backend
fix(engine): prevent OOM when VRAM < vram_gb threshold
docs(readme): add How it works diagram
chore: bump diffusers to 0.32.0
```

## Pull Requests

- Target the `staging` branch.
- Use Conventional Commits for the PR title (see above).
- Keep one logical change per PR. Link the related issue with `Closes #N`.
- Include a short description of what changed and why.
- All quality gates must pass before requesting review.

## Code Review

- Reviews focus on correctness, memory safety, and convention adherence — not style (ruff handles that).
- Expect at most one round of feedback for small PRs; larger changes may need more discussion.
- Reviewer may ask for a test if a bug fix has no coverage.
- All `ruff` and `pytest` failures must be resolved before merge.
