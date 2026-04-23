"""``imagecli engines`` and ``imagecli info`` commands."""

from __future__ import annotations

import typer
from rich.table import Table

from imagecli.commands._helpers import console, load_config


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


def info():
    """Show active config and GPU info."""
    cfg = load_config()

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


def register(app: typer.Typer) -> None:
    app.command()(engines)
    app.command()(info)
