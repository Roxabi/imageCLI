"""Shared helpers used by the CLI command modules.

Keeping these in one place avoids circular imports between command modules
and lets batch strategies reuse ``run_generate`` without pulling ``cli.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from imagecli.lora_spec import LoraSpec

console = Console()


def resolve_loras(
    cli_loras: list[str],
    cli_scales: list[float],
    cli_triggers: list[str],
    cli_embeddings: list[str],
    fm_loras: list[LoraSpec],
) -> list[LoraSpec]:
    """Resolve LoRA list from CLI flags and/or frontmatter.

    CLI non-empty → replaces FM list entirely (no merge).
    Empty CLI → return FM list as-is.
    Pairwise: scales/triggers/embeddings must be empty OR same length as loras.
    """
    if not cli_loras:
        return list(fm_loras)

    n = len(cli_loras)

    def _check_pairwise(values: list, name: str) -> None:
        if values and len(values) != n:
            raise ValueError(
                f"--{name} repeated {len(values)} time(s) but --lora repeated {n} time(s); "
                f"they must match (or omit --{name} entirely for defaults)."
            )

    _check_pairwise(cli_scales, "lora-scale")
    _check_pairwise(cli_triggers, "trigger")
    _check_pairwise(cli_embeddings, "embedding")

    result: list[LoraSpec] = []
    for i, path in enumerate(cli_loras):
        result.append(
            LoraSpec(
                path=path,
                scale=cli_scales[i] if cli_scales else 1.0,
                trigger=cli_triggers[i] if cli_triggers else None,
                embedding_path=cli_embeddings[i] if cli_embeddings else None,
            )
        )
    return result


def resolve_face_image(face_image: str | None, prompt_dir: Path) -> str | None:
    """Resolve face_image path: absolute → as-is, relative → relative to prompt file dir."""
    if not face_image:
        return None
    p = Path(face_image)
    return str(p if p.is_absolute() else (prompt_dir / p).resolve())


def resolve_face_images(face_images: list[str] | None, prompt_dir: Path) -> list[str] | None:
    """Resolve a list of face image paths the same way as resolve_face_image."""
    if not face_images:
        return None
    return [resolve_face_image(p, prompt_dir) for p in face_images]  # type: ignore[misc]


def load_config() -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from imagecli.config import load_config as _load_config

        cfg = _load_config()
    for w in caught:
        Console(stderr=True).print(f"[yellow]Warning:[/yellow] {w.message}")
    return cfg


def run_generate(
    prompt: str,
    negative_prompt: str,
    engine_name: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
    fmt: str,
    output_path: Path,
    engine_instance: object | None = None,
    *,
    compile: bool = True,
    negative_explicit: bool = False,
    steps_explicit: bool = False,
    guidance_explicit: bool = False,
    face_image: str | None = None,
    face_images: list[str] | None = None,
    pulid_strength: float = 0.6,
    loras: list[LoraSpec] | None = None,
):
    from imagecli.engine import ImageEngine, get_engine, preflight_check, warn_ignored_params

    engine: ImageEngine = engine_instance or get_engine(
        engine_name,
        compile=compile,
        loras=loras or [],
    )

    preflight_check(engine)
    warn_ignored_params(
        engine,
        negative_prompt,
        steps,
        guidance,
        negative_explicit=negative_explicit,
        steps_explicit=steps_explicit,
        guidance_explicit=guidance_explicit,
    )

    console.print(f"Engine: [bold cyan]{engine_name}[/bold cyan] — {engine.description}")
    console.print(f"Size: {width}×{height}  Steps: {steps}  Guidance: {guidance}")
    if seed is not None:
        console.print(f"Seed: {seed}")
    else:
        console.print("Seed: [dim]auto (saved in PNG metadata)[/dim]")
    console.print(f"Prompt: [italic]{prompt[:120]}{'…' if len(prompt) > 120 else ''}[/italic]")
    console.print(f"Output → [green]{output_path}[/green]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Generating…", total=engine.effective_steps(steps))

            def _step_callback(pipeline, step, timestep, callback_kwargs):  # noqa: ANN001, ANN202
                progress.update(task, completed=step + 1)
                return callback_kwargs

            saved = engine.generate(
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=seed,
                output_path=output_path,
                callback=_step_callback,
                face_image=face_image,
                face_images=face_images,
                pulid_strength=pulid_strength,
            )
    finally:
        if engine_instance is None:
            engine.cleanup()

    console.print(f"[bold green]Saved:[/bold green] {saved}")
    try:
        import torch

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_reserved() / 1024**3
            console.print(f"Peak VRAM reserved (torch): [cyan]{peak_gb:.2f} GB[/cyan]")
    except ImportError:
        pass
    return saved
