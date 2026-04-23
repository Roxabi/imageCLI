"""``imagecli batch`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from imagecli.commands._batch_all_on_gpu import run_all_on_gpu
from imagecli.commands._batch_sequential import run_sequential
from imagecli.commands._batch_two_phase import run_two_phase
from imagecli.commands._helpers import console, load_config


def batch(
    directory: Annotated[str, typer.Argument(help="Directory containing .md prompt files.")],
    engine: Annotated[Optional[str], typer.Option("-e", "--engine")] = None,
    output_dir: Annotated[Optional[str], typer.Option("--output-dir")] = None,
    no_compile: Annotated[
        bool,
        typer.Option("--no-compile", help="Skip torch.compile (faster startup, slower gen)."),
    ] = False,
    two_phase: Annotated[
        bool,
        typer.Option(
            "--two-phase",
            help=(
                "Force 2-phase batch (encode all, then generate all). "
                "Uses less VRAM (~8 GB peak) but slower."
            ),
        ),
    ] = False,
    steps: Annotated[
        Optional[int],
        typer.Option("--steps", "-s", help="Override inference steps for all prompts."),
    ] = None,
    lora: Annotated[
        Optional[str],
        typer.Option(
            "--lora", help="Path to LoRA weights (.safetensors). Loaded before quantization."
        ),
    ] = None,
    lora_scale: Annotated[
        Optional[float],
        typer.Option("--lora-scale", help="LoRA adapter scale (default 1.0)."),
    ] = None,
    trigger: Annotated[
        Optional[str],
        typer.Option(
            "--trigger",
            help=(
                "Pivotal tuning trigger word (e.g. 'lyraface'). Required when "
                "the LoRA was trained with ai-toolkit's 'embedding:' block. "
                "See docs/lora.md for details."
            ),
        ),
    ] = None,
    embedding: Annotated[
        Optional[str],
        typer.Option(
            "--embedding",
            help=(
                "Path to a standalone pivotal embedding (.safetensors). "
                "Overrides emb_params in the LoRA file. See docs/lora.md."
            ),
        ),
    ] = None,
):
    """Generate images for all .md files in a directory."""
    cfg = load_config()
    prompt_dir = Path(directory)
    files = sorted(prompt_dir.glob("*.md"))

    if not files:
        console.print(f"[yellow]No .md files found in {directory}[/yellow]")
        raise typer.Exit(1)

    console.print(f"Batch: {len(files)} prompt(s) found in {directory}")

    from imagecli.engine import get_engine
    from imagecli.markdown import parse_prompt_file

    parsed = []
    for f in files:
        doc = parse_prompt_file(f)
        parsed.append((f, doc, engine or doc.engine or cfg["engine"]))

    engine_names = {name for _, _, name in parsed}
    single_engine = len(engine_names) == 1

    if single_engine:
        the_engine_name = engine_names.pop()
        batch_lora = lora or parsed[0][1].lora_path
        batch_lora_scale = lora_scale if lora_scale is not None else parsed[0][1].lora_scale
        batch_trigger = trigger or parsed[0][1].trigger
        batch_embedding = embedding or parsed[0][1].embedding_path
        the_engine = get_engine(
            the_engine_name,
            compile=not no_compile,
            lora_path=batch_lora,
            lora_scale=batch_lora_scale,
            trigger=batch_trigger,
            embedding_path=batch_embedding,
        )

        if hasattr(the_engine, "load_all_on_gpu") and not two_phase:
            successes, failures = run_all_on_gpu(
                parsed,
                the_engine,
                the_engine_name,
                cfg,
                output_dir,
                console,
                steps_override=steps,
            )
        elif the_engine.supports_two_phase:
            successes, failures = run_two_phase(
                parsed,
                the_engine,
                the_engine_name,
                cfg,
                output_dir,
                no_compile,
                console,
                steps_override=steps,
            )
        else:
            successes, failures = run_sequential(
                parsed,
                cfg,
                output_dir,
                no_compile,
                console,
                initial_engine=(the_engine, the_engine_name),
                steps_override=steps,
            )
    else:
        successes, failures = run_sequential(
            parsed,
            cfg,
            output_dir,
            no_compile,
            console,
            steps_override=steps,
            lora_override=lora,
            lora_scale_override=lora_scale,
            trigger_override=trigger,
            embedding_override=embedding,
        )

    console.rule()
    console.print(f"Done: [green]{successes} succeeded[/green], [red]{failures} failed[/red]")


def register(app: typer.Typer) -> None:
    app.command()(batch)
