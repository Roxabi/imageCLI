"""imagecli — unified local image generation, usable as both CLI and library."""

from __future__ import annotations

import warnings
from pathlib import Path

from imagecli.config import load_config
from imagecli.engine import (
    ImageEngine,
    InsufficientResourcesError,
    get_engine,
    list_engines,
    preflight_check,
)
from imagecli.markdown import PromptDoc, parse_prompt_file

__all__ = [
    "ImageEngine",
    "InsufficientResourcesError",
    "PromptDoc",
    "generate",
    "get_engine",
    "list_engines",
    "load_config",
    "parse_prompt_file",
    "preflight_check",
]


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
) -> Path:
    """Generate an image from a text prompt. Returns the path to the saved image.

    All optional parameters default to values from ``imagecli.toml`` when not
    provided (``None``).  Priority: **kwarg > config > hardcoded default**.
    """
    from imagecli._utils import resolve_output

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config()

    engine_name = engine or cfg["engine"]
    w = width or cfg["width"]
    h = height or cfg["height"]
    s = steps or cfg["steps"]
    g = guidance or cfg["guidance"]
    fmt = format or cfg.get("format", "png")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = resolve_output(cfg, "image", fmt, str(output_dir) if output_dir else None)

    eng = get_engine(engine_name, compile=compile)
    preflight_check(eng)

    try:
        return eng.generate(
            prompt,
            negative_prompt=negative_prompt,
            width=w,
            height=h,
            steps=s,
            guidance=g,
            seed=seed,
            output_path=out,
        )
    finally:
        eng.cleanup()
