"""``imagecli serve`` and ``imagecli nats-serve image`` commands."""

from __future__ import annotations

import logging

import typer

log = logging.getLogger(__name__)

_nats_app = typer.Typer(help="NATS subscriber for Lyra-driven image generation.")


def serve(
    engine: str = typer.Option("flux2-klein", "--engine", "-e", help="Engine to preload"),
) -> None:
    """Start image generation daemon — keeps model warm in VRAM between requests."""
    from imagecli.daemon import daemon_main

    daemon_main(engine)


def _init_blob_store() -> object:
    """Instantiate HttpBlobStore from config (ADR-067, #97). Fail-fast on missing token.

    Returns the store as ``object`` to avoid leaking the roxabi_blobs import into
    callers that may not need it yet. Slice 1 of #97 only instantiates and probes;
    slice 2 wires the store into ``ImageNatsAdapter``.
    """
    from imagecli.config import load_blobstore_config
    from roxabi_blobs.http_store import HttpBlobStore

    cfg = load_blobstore_config()
    if cfg["token"] is None:
        raise RuntimeError(
            "blobstore token not configured — set imagecli.toml [blobstore].token, "
            "IMAGECLI_BLOBSTORE_TOKEN env var, or mount the Quadlet secret "
            "'imagecli-blobstore-token'."
        )
    endpoint = cfg["endpoint"]
    assert endpoint is not None  # load_blobstore_config guarantees a default
    return HttpBlobStore(base_url=endpoint, token=cfg["token"])


async def _probe_blobstore(endpoint: str) -> None:
    """Best-effort startup connectivity check. Warn on failure, never raise."""
    import httpx

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
        log.warning(
            "HttpBlobStore probe failed for %s: %s — first put() may fail",
            endpoint,
            exc,
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
    endpoint = cfg["endpoint"]
    assert endpoint is not None
    asyncio.run(_probe_blobstore(endpoint))

    from roxabi_blobs.protocol import BlobStore as _BlobStoreProtocol

    assert isinstance(blob_store, _BlobStoreProtocol)
    adapter = ImageNatsAdapter(default_engine=engine, blob_store=blob_store)
    try:
        asyncio.run(adapter.run(nats_url=nats_url))
    except KeyboardInterrupt:
        pass


def register(app: typer.Typer) -> None:
    app.command()(serve)
    app.add_typer(_nats_app, name="nats-serve")
