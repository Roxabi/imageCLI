"""imagecli — unified CLI for local image generation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, Optional

import typer
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
from rich.table import Table

from imagecli._utils import resolve_output as _resolve_output


def _resolve_face_image(face_image: str | None, prompt_dir: Path) -> str | None:
    """Resolve face_image path: absolute → as-is, relative → relative to prompt file dir."""
    if not face_image:
        return None
    p = Path(face_image)
    return str(p if p.is_absolute() else (prompt_dir / p).resolve())

app = typer.Typer(
    name="imagecli",
    help="Local image generation — FLUX.2-klein, FLUX.1-dev, SD3.5 backends.",
    no_args_is_help=True,
)
console = Console()


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_config() -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from imagecli.config import load_config

        cfg = load_config()
    for w in caught:
        Console(stderr=True).print(f"[yellow]Warning:[/yellow] {w.message}")
    return cfg


def _run_generate(
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
    pulid_strength: float = 0.6,
):
    from imagecli.engine import ImageEngine, get_engine, preflight_check, warn_ignored_params

    engine: ImageEngine = engine_instance or get_engine(engine_name, compile=compile)

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


# ── commands ─────────────────────────────────────────────────────────────────


@app.command()
def generate(
    prompt_or_file: Annotated[str, typer.Argument(help="Prompt text or path to a .md file.")],
    engine: Annotated[Optional[str], typer.Option("-e", "--engine", help="Engine name.")] = None,
    width: Annotated[Optional[int], typer.Option("-W", "--width")] = None,
    height: Annotated[Optional[int], typer.Option("-H", "--height")] = None,
    steps: Annotated[Optional[int], typer.Option("-s", "--steps")] = None,
    guidance: Annotated[Optional[float], typer.Option("-g", "--guidance")] = None,
    seed: Annotated[Optional[int], typer.Option("--seed")] = None,
    negative: Annotated[str, typer.Option("-n", "--negative", help="Negative prompt.")] = "",
    output: Annotated[
        Optional[str], typer.Option("-o", "--output", help="Output file path.")
    ] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir")] = None,
    fmt: Annotated[str, typer.Option("--format", help="Output format: png | jpg | webp")] = "png",
    face_image: Annotated[
        Optional[str],
        typer.Option("--face-image", help="Reference face image for PuLID engines."),
    ] = None,
    pulid_strength: Annotated[
        Optional[float],
        typer.Option("--pulid-strength", help="PuLID identity lock strength (default 0.6)."),
    ] = None,
    no_compile: Annotated[
        bool, typer.Option("--no-compile", help="Skip torch.compile (faster startup, slower gen).")
    ] = False,
):
    """Generate an image from a prompt or .md file."""
    cfg = _load_config()

    path = Path(prompt_or_file)
    if path.suffix == ".md" and path.exists():
        from imagecli.markdown import parse_prompt_file

        doc = parse_prompt_file(path)
        prompt_text = doc.prompt
        stem = path.stem
        # frontmatter overrides config; CLI flags override frontmatter
        engine_name = engine or doc.engine or cfg["engine"]
        w = width or doc.width or cfg["width"]
        h = height or doc.height or cfg["height"]
        s = steps or doc.steps or cfg["steps"]
        g = guidance or doc.guidance or cfg["guidance"]
        sd = seed if seed is not None else doc.seed
        neg = negative or doc.negative_prompt
        out_fmt = fmt or doc.format or cfg.get("format", "png")
        negative_explicit = bool(negative) or bool(doc.negative_prompt)
        steps_explicit = steps is not None or doc.steps is not None
        guidance_explicit = guidance is not None or doc.guidance is not None
        face_img = (
            _resolve_face_image(face_image, Path.cwd())
            or _resolve_face_image(doc.face_image, path.parent)
        )
        pulid_str = pulid_strength if pulid_strength is not None else doc.pulid_strength
    else:
        prompt_text = prompt_or_file
        stem = "image"
        engine_name = engine or cfg["engine"]
        w = width or cfg["width"]
        h = height or cfg["height"]
        s = steps or cfg["steps"]
        g = guidance or cfg["guidance"]
        sd = seed
        neg = negative
        out_fmt = fmt or cfg.get("format", "png")
        negative_explicit = bool(negative)
        steps_explicit = steps is not None
        guidance_explicit = guidance is not None
        face_img = _resolve_face_image(face_image, Path.cwd())
        pulid_str = pulid_strength if pulid_strength is not None else 0.6

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = _resolve_output(cfg, stem, out_fmt, output_dir)

    _run_generate(
        prompt_text,
        neg,
        engine_name,
        w,
        h,
        s,
        g,
        sd,
        out_fmt,
        out_path,
        compile=not no_compile,
        negative_explicit=negative_explicit,
        steps_explicit=steps_explicit,
        guidance_explicit=guidance_explicit,
        face_image=face_img,
        pulid_strength=pulid_str,
    )


@app.command()
def batch(
    directory: Annotated[str, typer.Argument(help="Directory containing .md prompt files.")],
    engine: Annotated[Optional[str], typer.Option("-e", "--engine")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir")] = None,
    no_compile: Annotated[
        bool, typer.Option("--no-compile", help="Skip torch.compile (faster startup, slower gen).")
    ] = False,
):
    """Generate images for all .md files in a directory."""
    cfg = _load_config()
    prompt_dir = Path(directory)
    files = sorted(prompt_dir.glob("*.md"))

    if not files:
        console.print(f"[yellow]No .md files found in {directory}[/yellow]")
        raise typer.Exit(1)

    console.print(f"Batch: {len(files)} prompt(s) found in {directory}")

    from imagecli.engine import ImageEngine, get_engine
    from imagecli.markdown import parse_prompt_file

    # Track the currently loaded engine so we can reuse it when consecutive
    # prompts share the same engine, and fully unload when they don't.
    current_engine: ImageEngine | None = None
    current_engine_name: str | None = None

    n_files = len(files)
    successes, failures = 0, 0
    for i, f in enumerate(files):
        console.rule(f"[bold]{i + 1}/{n_files}[/bold] — {f.name}")
        try:
            doc = parse_prompt_file(f)
            engine_name = engine or doc.engine or cfg["engine"]

            # If the engine changed, fully unload the previous one first.
            if current_engine is not None and engine_name != current_engine_name:
                console.print(
                    f"[dim]Engine changed ({current_engine_name} → {engine_name}), "
                    f"unloading previous…[/dim]"
                )
                current_engine.cleanup()
                current_engine = None
                current_engine_name = None

            if current_engine is None:
                current_engine = get_engine(engine_name, compile=not no_compile)
                current_engine_name = engine_name

            w = doc.width or cfg["width"]
            h = doc.height or cfg["height"]
            s = doc.steps or cfg["steps"]
            g = doc.guidance or cfg["guidance"]
            fmt = doc.format or cfg.get("format", "png")
            out_path = _resolve_output(cfg, f.stem, fmt, output_dir)
            negative_explicit = bool(doc.negative_prompt)
            steps_explicit = doc.steps is not None
            guidance_explicit = doc.guidance is not None
            _run_generate(
                doc.prompt,
                doc.negative_prompt,
                engine_name,
                w,
                h,
                s,
                g,
                doc.seed,
                fmt,
                out_path,
                engine_instance=current_engine,
                negative_explicit=negative_explicit,
                steps_explicit=steps_explicit,
                guidance_explicit=guidance_explicit,
                face_image=_resolve_face_image(doc.face_image, f.parent),
                pulid_strength=doc.pulid_strength,
            )
            successes += 1

            # Clear CUDA cache between images to prevent VRAM fragmentation.
            current_engine.clear_cache()

        except Exception as e:
            console.print(f"[red]FAILED:[/red] {e}")
            failures += 1
            # On failure, fully unload to recover VRAM/RAM for the next image.
            if current_engine is not None:
                current_engine.cleanup()
                current_engine = None
                current_engine_name = None

    # Final cleanup
    if current_engine is not None:
        current_engine.cleanup()

    console.rule()
    console.print(f"Done: [green]{successes} succeeded[/green], [red]{failures} failed[/red]")


@app.command()
def engines():
    """List available image generation engines."""
    from imagecli.engine import list_engines

    table = Table(title="Available Engines")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("VRAM", justify="right")
    table.add_column("Neg Prompt", justify="center")
    table.add_column("Steps", justify="center")
    table.add_column("Guidance", justify="center")
    table.add_column("Description")
    table.add_column("Model ID", style="dim")

    for e in list_engines():
        caps = e["capabilities"]
        neg = "✓" if caps["negative_prompt"] else "✗"
        steps_col = f"fixed {caps['fixed_steps']}" if caps["fixed_steps"] is not None else "✓"
        guidance_col = (
            f"fixed {caps['fixed_guidance']}" if caps["fixed_guidance"] is not None else "✓"
        )
        table.add_row(
            e["name"],
            f"{e['vram_gb']}GB",
            neg,
            steps_col,
            guidance_col,
            e["description"],
            e["model_id"],
        )

    console.print(table)


@app.command()
def info():
    """Show active config and GPU info."""
    cfg = _load_config()

    table = Table(title="imagecli config")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for k, v in cfg.items():
        table.add_row(str(k), str(v))
    console.print(table)

    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            vram_gb = dev.total_memory / 1024**3
            sm = f"sm_{dev.major}{dev.minor}"
            sm_tuple = (dev.major, dev.minor)
            arch = (
                "Blackwell"
                if sm_tuple >= (12, 0)
                else "Hopper"
                if sm_tuple >= (9, 0)
                else "Ada Lovelace"
                if sm_tuple >= (8, 9)
                else "Ampere"
                if sm_tuple >= (8, 0)
                else "Turing"
                if sm_tuple >= (7, 0)
                else "Pascal"
                if sm_tuple >= (6, 0)
                else "Unknown"
            )
            console.print(f"\nGPU: [bold]{dev.name}[/bold] — {vram_gb:.1f} GB VRAM — {sm} ({arch})")
        else:
            console.print("\n[yellow]No CUDA GPU detected.[/yellow]")
    except ImportError:
        console.print("\n[yellow]torch not installed — cannot detect GPU.[/yellow]")
