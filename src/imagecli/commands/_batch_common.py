"""Shared progress/console machinery for batch strategies."""

from __future__ import annotations

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def progress_bar(console_):
    """Standard Progress instance used by all batch strategies."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console_,
        transient=True,
    )


def report_peak_vram(console_) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_reserved() / 1024**3
            console_.print(f"Peak VRAM reserved (torch): [cyan]{peak_gb:.2f} GB[/cyan]")
    except ImportError:
        pass


def resolve_image_params(doc, cfg, steps_override):
    """Common per-image config resolution used by the GPU-resident strategies."""
    return {
        "w": doc.width or cfg["width"],
        "h": doc.height or cfg["height"],
        "s": steps_override or doc.steps or cfg["steps"],
        "g": doc.guidance or cfg["guidance"],
        "fmt": doc.format or cfg.get("format", "png"),
    }
