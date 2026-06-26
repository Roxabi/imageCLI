"""``imagecli serve`` and ``imagecli nats-serve image`` commands."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from roxabi_blobs.protocol import BlobStore

log = logging.getLogger(__name__)

_nats_app = typer.Typer(help="NATS subscriber for Lyra-driven image generation.")


def serve(
    engine: str = typer.Option("flux2-klein", "--engine", "-e", help="Engine to preload"),
) -> None:
    """Start image generation daemon — keeps model warm in VRAM between requests."""
    from imagecli.daemon import daemon_main

    daemon_main(engine)


def _init_blob_store() -> BlobStore:
    """Return the shared HttpBlobStore singleton (ADR-067, #97)."""
    from imagecli.nats.blobs import get_blobstore

    return get_blobstore()


async def _probe_blobstore(endpoint: str) -> None:
    """Best-effort startup connectivity check. Warn on failure, never raise."""
    import httpx

    from imagecli.nats.validators import _sanitize_delivery_exception

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{endpoint}/healthz")
            if resp.status_code >= 400:
                log.warning(
                    "HttpBlobStore probe: %s/healthz returned %s — first put() may fail",
                    endpoint,
                    resp.status_code,
                )
    except Exception as exc:  # noqa: BLE001 — probe is best-effort, must not raise
        # Sanitize: httpx exception strings can carry URLs / auth context. Endpoint
        # is operator-set and safe to log; exc class is conveyed via the helper.
        log.warning(
            "HttpBlobStore probe failed for %s: %s — first put() may fail",
            endpoint,
            _sanitize_delivery_exception(exc),
        )


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

    from imagecli.config import load_blobstore_config
    from imagecli.nats import ImageNatsAdapter

    # ADR-067 (#97): instantiate the cross-host BlobStore, run the warn-only
    # connectivity probe, then wire the store into the adapter so handle()
    # emits ImageResponse.blob_ref instead of inline bytes.
    blob_store = _init_blob_store()
    cfg = load_blobstore_config()
    asyncio.run(_probe_blobstore(cfg["endpoint"]))

    adapter = ImageNatsAdapter(default_engine=engine, blob_store=blob_store)
    try:
        asyncio.run(adapter.run(nats_url=nats_url))
    except KeyboardInterrupt:
        pass


def register(app: typer.Typer) -> None:
    app.command()(serve)
    app.add_typer(_nats_app, name="nats-serve")
