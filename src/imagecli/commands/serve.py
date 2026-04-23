"""``imagecli serve`` and ``imagecli nats-serve image`` commands."""

from __future__ import annotations

import typer

_nats_app = typer.Typer(help="NATS subscriber for Lyra-driven image generation.")


def serve(
    engine: str = typer.Option("flux2-klein", "--engine", "-e", help="Engine to preload"),
) -> None:
    """Start image generation daemon — keeps model warm in VRAM between requests."""
    from imagecli.daemon import daemon_main

    daemon_main(engine)


@_nats_app.command("image")
def nats_serve(
    engine: str = typer.Option(
        "flux2-klein", "--engine", "-e", help="Default engine for requests."
    ),
    nats_url: str = typer.Option(
        "nats://localhost:4222", envvar="NATS_URL", help="NATS server URL."
    ),
) -> None:
    """Start NATS subscriber for image generation requests."""
    import asyncio

    from imagecli.nats import ImageNatsAdapter

    adapter = ImageNatsAdapter(default_engine=engine)
    try:
        asyncio.run(adapter.run(nats_url=nats_url))
    except KeyboardInterrupt:
        pass


def register(app: typer.Typer) -> None:
    app.command()(serve)
    app.add_typer(_nats_app, name="nats-serve")
