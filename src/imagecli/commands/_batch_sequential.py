"""Sequential batch strategy: one engine at a time, CPU offload per image."""

from __future__ import annotations

from imagecli._utils import resolve_output
from imagecli.commands._helpers import (
    resolve_face_image,
    resolve_face_images,
    run_generate,
)


def run_sequential(
    parsed,
    cfg,
    output_dir,
    no_compile,
    console_,
    initial_engine=None,
    steps_override=None,
    lora_override=None,
    lora_scale_override=None,
    trigger_override=None,
    embedding_override=None,
):
    from imagecli.engine import ImageEngine, get_engine

    if initial_engine is not None:
        current_engine, current_engine_name = initial_engine
    else:
        current_engine: ImageEngine | None = None
        current_engine_name: str | None = None

    n_files = len(parsed)
    successes, failures = 0, 0
    for i, (f, doc, engine_name) in enumerate(parsed):
        console_.rule(f"[bold]{i + 1}/{n_files}[/bold] — {f.name}")
        try:
            if current_engine is not None and engine_name != current_engine_name:
                console_.print(
                    f"[dim]Engine changed ({current_engine_name} → {engine_name}), "
                    f"unloading previous…[/dim]"
                )
                current_engine.cleanup()
                current_engine = None
                current_engine_name = None

            if current_engine is None:
                current_engine = get_engine(
                    engine_name,
                    compile=not no_compile,
                    lora_path=lora_override or doc.lora_path,
                    lora_scale=lora_scale_override
                    if lora_scale_override is not None
                    else doc.lora_scale,
                    trigger=trigger_override or doc.trigger,
                    embedding_path=embedding_override or doc.embedding_path,
                )
                current_engine_name = engine_name

            w = doc.width or cfg["width"]
            h = doc.height or cfg["height"]
            s = steps_override or doc.steps or cfg["steps"]
            g = doc.guidance or cfg["guidance"]
            fmt = doc.format or cfg.get("format", "png")
            out_path = resolve_output(cfg, f.stem, fmt, output_dir)
            negative_explicit = bool(doc.negative_prompt)
            steps_explicit = doc.steps is not None
            guidance_explicit = doc.guidance is not None
            run_generate(
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
                face_image=resolve_face_image(doc.face_image, f.parent),
                face_images=resolve_face_images(doc.face_images, f.parent),
                pulid_strength=doc.pulid_strength,
            )
            successes += 1

            current_engine.clear_cache()

        except Exception as e:
            console_.print(f"[red]FAILED:[/red] {e}")
            failures += 1
            if current_engine is not None:
                current_engine.cleanup()
                current_engine = None
                current_engine_name = None

    if current_engine is not None:
        current_engine.cleanup()

    return successes, failures
