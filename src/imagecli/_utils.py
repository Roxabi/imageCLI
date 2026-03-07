"""Shared utility functions for imagecli."""

from __future__ import annotations

from pathlib import Path


def resolve_output(cfg: dict, stem: str, fmt: str, output_dir: str | None) -> Path:
    """Resolve output path with anti-clobber suffixing (_1, _2, etc.)."""
    out_dir = Path(output_dir or cfg.get("output_dir", "images/images_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = fmt.lstrip(".")
    candidate = out_dir / f"{stem}.{ext}"
    n = 1
    while candidate.exists():
        candidate = out_dir / f"{stem}_{n}.{ext}"
        n += 1
    return candidate
