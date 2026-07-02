"""All-on-GPU batch strategy: encoder + transformer + VAE resident (~12 GB)."""

from __future__ import annotations

from imagecli._utils import resolve_output
from imagecli.commands._batch_common import (
    progress_bar,
    report_peak_vram,
    resolve_image_params,
)


def run_all_on_gpu(parsed, engine, engine_name, cfg, output_dir, console_, steps_override=None):
    from imagecli.engine import preflight_check

    n_files = len(parsed)
    successes, failures = 0, 0

    try:
        preflight_check(engine)

        console_.print(
            f"\n[bold cyan]All-on-GPU batch:[/bold cyan] {n_files} images "
            f"(encoder + transformer + VAE, ~12 GB)..."
        )
        engine.load_all_on_gpu()

        for i, (f, doc, _) in enumerate(parsed):
            console_.rule(f"[bold]{i + 1}/{n_files}[/bold] — {f.name}")
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

                    saved = engine.encode_and_generate(
                        doc.prompt,
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
