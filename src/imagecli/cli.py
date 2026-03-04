"""imagecli — unified CLI for local image generation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

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


def _resolve_output(cfg: dict, stem: str, fmt: str, output_dir: str | None) -> Path:
    out_dir = Path(output_dir or cfg.get("output_dir", "images/images_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = fmt.lstrip(".")
    # avoid clobbering: add suffix if file exists
    candidate = out_dir / f"{stem}.{ext}"
    n = 1
    while candidate.exists():
        candidate = out_dir / f"{stem}_{n}.{ext}"
        n += 1
    return candidate


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
) -> Path:
    from imagecli.engine import get_engine

    engine = get_engine(engine_name)
    console.print(f"Engine: [bold cyan]{engine_name}[/bold cyan] — {engine.description}")
    console.print(f"Size: {width}×{height}  Steps: {steps}  Guidance: {guidance}")
    if seed is not None:
        console.print(f"Seed: {seed}")
    console.print(f"Prompt: [italic]{prompt[:120]}{'…' if len(prompt) > 120 else ''}[/italic]")
    console.print(f"Output → [green]{output_path}[/green]")

    saved = engine.generate(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed,
        output_path=output_path,
    )
    console.print(f"\n[bold green]Saved:[/bold green] {saved}")
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
):
    """Generate an image from a prompt or .md file."""
    cfg = _load_config()

    path = Path(prompt_or_file)
    if path.suffix == ".md" and path.exists():
        from imagecli.markdown import parse_prompt_file

        doc = parse_prompt_file(path)
        prompt_text = doc.prompt
        stem = path.stem
        # priority: CLI flag > frontmatter > toml > engine default
        engine_name = engine or doc.engine or cfg["engine"]
        from imagecli.engine import get_engine_class

        ecls = get_engine_class(engine_name)
        w = width or doc.width or cfg["width"]
        h = height or doc.height or cfg["height"]
        s = steps or doc.steps or cfg.get("steps") or ecls.default_steps
        g = guidance or doc.guidance or cfg.get("guidance") or ecls.default_guidance
        sd = seed if seed is not None else doc.seed
        neg = negative or doc.negative_prompt
        out_fmt = fmt or doc.format or cfg.get("format", "png")
    else:
        prompt_text = prompt_or_file
        stem = "image"
        engine_name = engine or cfg["engine"]
        from imagecli.engine import get_engine_class

        ecls = get_engine_class(engine_name)
        w = width or cfg["width"]
        h = height or cfg["height"]
        s = steps or cfg.get("steps") or ecls.default_steps
        g = guidance or cfg.get("guidance") or ecls.default_guidance
        sd = seed
        neg = negative
        out_fmt = fmt or cfg.get("format", "png")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = _resolve_output(cfg, stem, out_fmt, output_dir)

    _run_generate(prompt_text, neg, engine_name, w, h, s, g, sd, out_fmt, out_path)


@app.command()
def batch(
    directory: Annotated[str, typer.Argument(help="Directory containing .md prompt files.")],
    engine: Annotated[Optional[str], typer.Option("-e", "--engine")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir")] = None,
):
    """Generate images for all .md files in a directory."""
    cfg = _load_config()
    prompt_dir = Path(directory)
    files = sorted(prompt_dir.glob("*.md"))

    if not files:
        console.print(f"[yellow]No .md files found in {directory}[/yellow]")
        raise typer.Exit(1)

    console.print(f"Batch: {len(files)} prompt(s) found in {directory}")

    from imagecli.markdown import parse_prompt_file

    successes, failures = 0, 0
    for f in files:
        console.rule(f.name)
        try:
            doc = parse_prompt_file(f)
            engine_name = engine or doc.engine or cfg["engine"]
            from imagecli.engine import get_engine_class

            ecls = get_engine_class(engine_name)
            w = doc.width or cfg["width"]
            h = doc.height or cfg["height"]
            s = doc.steps or cfg.get("steps") or ecls.default_steps
            g = doc.guidance or cfg.get("guidance") or ecls.default_guidance
            fmt = doc.format or cfg.get("format", "png")
            out_path = _resolve_output(cfg, f.stem, fmt, output_dir)
            _run_generate(
                doc.prompt, doc.negative_prompt, engine_name, w, h, s, g, doc.seed, fmt, out_path
            )
            successes += 1
        except Exception as e:
            console.print(f"[red]FAILED:[/red] {e}")
            failures += 1

    console.rule()
    console.print(f"Done: [green]{successes} succeeded[/green], [red]{failures} failed[/red]")


@app.command()
def engines():
    """List available image generation engines."""
    from imagecli.engine import list_engines

    table = Table(title="Available Engines")
    table.add_column("Name", style="cyan")
    table.add_column("VRAM", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Guidance", justify="right")
    table.add_column("Description")
    table.add_column("Model ID", style="dim")

    for e in list_engines():
        table.add_row(
            e["name"],
            f"{e['vram_gb']}GB",
            str(e["default_steps"]),
            str(e["default_guidance"]),
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
            console.print(f"\nGPU: [bold]{dev.name}[/bold] — {vram_gb:.1f} GB VRAM")
        else:
            console.print("\n[yellow]No CUDA GPU detected.[/yellow]")
    except ImportError:
        console.print("\n[yellow]torch not installed — cannot detect GPU.[/yellow]")
