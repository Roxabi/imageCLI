"""2-phase batch strategy: encode all prompts, then generate all images."""

from __future__ import annotations

from imagecli._utils import resolve_output
from imagecli.commands._batch_common import (
    progress_bar,
    report_peak_vram,
    resolve_image_params,
)


def run_two_phase(
    parsed, engine, engine_name, cfg, output_dir, no_compile, console_, steps_override=None
):
    from imagecli.engine import preflight_check

    n_files = len(parsed)
    successes, failures = 0, 0

    try:
        preflight_check(engine)

        console_.print(
            f"\n[bold cyan]Phase 1/2:[/bold cyan] Encoding {n_files} prompts "
            f"(text encoder on GPU, ~8 GB)..."
        )
        engine.load_for_encode()

        encoded = []
        for i, (f, doc, _) in enumerate(parsed):
            try:
                emb = engine.encode_prompt(doc.prompt)
                encoded.append((f, doc, emb))
                console_.print(f"  [dim]{i + 1}/{n_files}[/dim] {f.name} [green]✓[/green]")
            except Exception as e:
                console_.print(f"  [dim]{i + 1}/{n_files}[/dim] {f.name} [red]✗ {e}[/red]")
                failures += 1

        console_.print(
            f"\n[bold cyan]Phase 2/2:[/bold cyan] Generating {len(encoded)} images "
            f"(transformer + VAE on GPU, ~4 GB)..."
        )
        engine.start_generation_phase()

        for i, (f, doc, emb) in enumerate(encoded):
            console_.rule(f"[bold]{i + 1}/{len(encoded)}[/bold] — {f.name}")
            try:
                p = resolve_image_params(doc, cfg, steps_override)
                out_path = resolve_output(cfg, f.stem, p["fmt"], output_dir)

                console_.print(f"Engine: [bold cyan]{engine_name}[/bold cyan]")
                console_.print(f"Size: {p['w']}×{p['h']}  Steps: {p['s']}  Guidance: {p['g']}")
                if doc.seed is not None:
                    console_.print(f"Seed: {doc.seed}")
                console_.print(f"Output → [green]{out_path}[/green]")

                with progress_bar(console_) as progress:
                    task = progress.add_task("Generating…", total=engine.effective_steps(p["s"]))

                    def _step_callback(pipeline, step, timestep, callback_kwargs):
                        progress.update(task, completed=step + 1)
                        return callback_kwargs

                    saved = engine.generate_from_embeddings(
                        emb,
                        width=p["w"],
                        height=p["h"],
                        steps=p["s"],
                        guidance=p["g"],
                        seed=doc.seed,
                        output_path=out_path,
                        callback=_step_callback,
                    )

                console_.print(f"[bold green]Saved:[/bold green] {saved}")
                report_peak_vram(console_)

                successes += 1
                engine.clear_cache()

            except Exception as e:
                console_.print(f"[red]FAILED:[/red] {e}")
                failures += 1

    finally:
        engine.cleanup()

    return successes, failures
