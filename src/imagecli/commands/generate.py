"""``imagecli generate`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, List, Optional

import typer

from imagecli._utils import resolve_output
from imagecli.commands._helpers import (
    load_config,
    resolve_face_image,
    resolve_face_images,
    resolve_loras,
    run_generate,
)


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
    lora: Annotated[
        Optional[List[str]],
        typer.Option(
            "--lora", help="Path to LoRA weights (.safetensors). Repeatable for multi-LoRA."
        ),
    ] = None,
    lora_scale: Annotated[
        Optional[List[float]],
        typer.Option("--lora-scale", help="LoRA adapter scale (default 1.0). Pairwise with --lora."),
    ] = None,
    trigger: Annotated[
        Optional[List[str]],
        typer.Option(
            "--trigger",
            help=(
                "Pivotal tuning trigger word (e.g. 'lyraface'). Pairwise with --lora. "
                "Required when the LoRA was trained with ai-toolkit's 'embedding:' block. "
                "See docs/lora.md for details."
            ),
        ),
    ] = None,
    embedding: Annotated[
        Optional[List[str]],
        typer.Option(
            "--embedding",
            help=(
                "Path to a standalone pivotal embedding (.safetensors). Pairwise with --lora. "
                "Overrides emb_params in the LoRA file. See docs/lora.md."
            ),
        ),
    ] = None,
    no_compile: Annotated[
        bool,
        typer.Option("--no-compile", help="Skip torch.compile (faster startup, slower gen)."),
    ] = False,
):
    """Generate an image from a prompt or .md file."""
    cfg = load_config()

    cli_loras: list[str] = lora or []
    cli_scales: list[float] = lora_scale or []
    cli_triggers: list[str] = trigger or []
    cli_embeddings: list[str] = embedding or []

    path = Path(prompt_or_file)
    if path.suffix == ".md" and path.exists():
        from imagecli.markdown import parse_prompt_file

        doc = parse_prompt_file(path)
        prompt_text = doc.prompt
        stem = path.stem
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
        face_img = resolve_face_image(face_image, Path.cwd()) or resolve_face_image(
            doc.face_image, path.parent
        )
        face_imgs = resolve_face_images(doc.face_images, path.parent)
        pulid_str = pulid_strength if pulid_strength is not None else doc.pulid_strength
        fm_loras = getattr(doc, "loras", [])
        resolved_loras = resolve_loras(cli_loras, cli_scales, cli_triggers, cli_embeddings, fm_loras)
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
        face_img = resolve_face_image(face_image, Path.cwd())
        face_imgs = None
        pulid_str = pulid_strength if pulid_strength is not None else 0.6
        resolved_loras = resolve_loras(cli_loras, cli_scales, cli_triggers, cli_embeddings, [])

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = resolve_output(cfg, stem, out_fmt, output_dir)

    run_generate(
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
        face_images=face_imgs,
        pulid_strength=pulid_str,
        loras=resolved_loras,
    )


def register(app: typer.Typer) -> None:
    app.command()(generate)
