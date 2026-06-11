"""ImageNatsAdapter — imageCLI NATS satellite for async image generation requests."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roxabi_blobs.protocol import BlobStore
from roxabi_contracts.blob_ref import BlobRef as WireBlobRef
from roxabi_contracts.envelope import CONTRACT_VERSION
from roxabi_contracts.errors import WorkerError
from roxabi_contracts.image import SUBJECTS, ImageResponse
from roxabi_nats import NatsAdapterBase

from imagecli.nats.validators import (
    _map_exception_to_error,
    _resolve_loras,
    _sanitize_delivery_exception,
    _validate_request,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Map legacy free-text error codes (kept for backward compat in `error: str`)
# to structured WorkerError fields. `default_retryable` follows KNOWN_CODES
# in roxabi_contracts.errors. Unmapped codes fall back to worker.internal.
#
# `delivery_failed` covers an OSError moving the generated file into
# nats_out/ — generation succeeded but persistence failed. There is no
# `image.delivery_failed` in KNOWN_CODES yet (request upstream), so we use
# `worker.internal` per its docstring: "not covered by a more specific code."
# Hub routing on `retryable=True` works identically to `worker.crash`.
_WORKER_ERROR_MAP: dict[str, tuple[str, bool]] = {
    "missing_required_field": ("worker.validation", False),
    "unknown_engine": ("worker.validation", False),
    "engine_load_failed": ("image.engine_unavailable", True),
    "insufficient_resources": ("worker.capacity", True),
    "generation_failed": ("worker.crash", True),
    "delivery_failed": ("worker.internal", True),
}


def _make_worker_error(code: str, detail: str | None = None) -> WorkerError:
    """Build a structured WorkerError from a legacy free-text error code."""
    canonical, retryable = _WORKER_ERROR_MAP.get(code, ("worker.internal", True))
    return WorkerError(code=canonical, message=code, retryable=retryable, detail=detail)


class ImageNatsAdapter(NatsAdapterBase):
    """NATS adapter for image generation requests from Lyra hub.

    Subscribes to `factory.image.generate.request` and responds with generated images.
    Follows the satellite-bootstrap pattern established by voice adapters (ADR-039).
    """

    def __init__(
        self,
        default_engine: str = "flux2-klein",
        *,
        blob_store: BlobStore,
        max_concurrent: int = 1,
        heartbeat_interval: float = 5.0,
        drain_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            subject=SUBJECTS.image_request,
            queue_group="IMAGE_WORKERS",
            envelope_name="image",
            schema_version=SCHEMA_VERSION,
            heartbeat_subject=SUBJECTS.image_heartbeat,
            heartbeat_interval=heartbeat_interval,
            drain_timeout=drain_timeout,
            inbox_prefix="_inbox.imagecli-image",
            wait_ready=False,  # worker semantics — see NatsAdapterBase docstring
        )
        self.default_engine = default_engine
        # ADR-067: every successful image reply carries a BlobRef built by put().
        self._blob_store: BlobStore = blob_store
        self.max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._engine_loaded: str | None = None
        self._engine_instance: Any = None  # Cached engine for reuse

    async def handle(self, msg: Any, payload: dict) -> None:
        """Process an image generation request per ADR-046 contract."""
        request_id = payload.get("request_id", "")
        # trace_id is required min_length=1 on the response envelope; fall back
        # to request_id (then "unknown") so error paths can still produce a
        # valid ImageResponse when the client omits it.
        trace_id = payload.get("trace_id") or request_id or "unknown"
        job_id = payload.get("job_id")

        # Acquire semaphore for concurrency control
        async with self._sem:
            # Validate required fields and bounds
            valid, error = _validate_request(payload)
            if not valid:
                await self._reply_error(
                    msg, trace_id, request_id, "missing_required_field", error, job_id=job_id
                )
                return

            engine_name = payload["engine"]
            prompt = payload["prompt"]

            # Import engine module lazily (heavy torch imports)
            try:
                from imagecli.engine import get_engine, preflight_check, list_engines
            except ImportError as e:
                await self._reply_error(
                    msg,
                    trace_id,
                    request_id,
                    "engine_load_failed",
                    f"Import error: {e}",
                    job_id=job_id,
                )
                return

            # Validate engine name against registry
            engine_names = {e["name"] for e in list_engines()}
            if engine_name not in engine_names:
                await self._reply_error(
                    msg, trace_id, request_id, "unknown_engine", engine_name, job_id=job_id
                )
                return

            # Get or create engine instance.
            # No-LoRA / no-pivotal requests share a cached instance via
            # ModelRegistry (warm across requests, LRU-evicted on VRAM
            # pressure). Per-request LoRA/pivotal configs take a fresh
            # engine (registry keys by name only) and cleanup() after use.
            loras = _resolve_loras(payload)
            uses_per_request_cfg = bool(loras)

            try:
                if uses_per_request_cfg:
                    engine = get_engine(
                        engine_name,
                        compile=True,
                        loras=loras,
                    )
                else:
                    from imagecli.model_registry import model_registry

                    engine = model_registry.get(engine_name)
                self._engine_loaded = engine_name
            except Exception as e:
                error_code, error_detail = _map_exception_to_error(e)
                await self._reply_error(
                    msg, trace_id, request_id, error_code, error_detail, job_id=job_id
                )
                return

            # Preflight check (VRAM/RAM validation)
            try:
                preflight_check(engine)
            except Exception as e:
                error_code, error_detail = _map_exception_to_error(e)
                await self._reply_error(
                    msg, trace_id, request_id, error_code, error_detail, job_id=job_id
                )
                return

            # Extract generation params with defaults
            width = payload.get("width", 1024)
            height = payload.get("height", 1024)
            steps = payload.get("steps", 50)
            guidance = payload.get("guidance", 4.0)
            seed = payload.get("seed")
            negative_prompt = payload.get("negative_prompt", "")
            fmt = payload.get("format", "png")

            # Generate image
            start_time = time.monotonic()
            tmp_path: Path | None = None  # initialized for static analysis (W7 cleanup path)
            try:
                # Generate to a temp file first
                from tempfile import NamedTemporaryFile

                with NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                saved_path = engine.generate(
                    prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    output_path=tmp_path,
                )

                duration = time.monotonic() - start_time
                log.info(f"Generated {saved_path} in {duration:.2f}s with engine {engine_name}")

            except Exception as e:
                error_code, error_detail = _map_exception_to_error(e)
                log.exception(f"Generation failed: {e}")
                if uses_per_request_cfg:
                    engine.cleanup()
                else:
                    from imagecli.model_registry import model_registry

                    # Evict so the registry can reload cleanly next request.
                    model_registry.evict(engine_name)
                # Best-effort cleanup of the temp file (engine may have created it partial).
                if tmp_path is not None:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass
                await self._reply_error(
                    msg, trace_id, request_id, error_code, error_detail, job_id=job_id
                )
                return

            # PUT → build response → reply. Wrapped in a single try (#97 N2): any
            # failure between the PUT and the NATS reply lands as `delivery_failed`
            # rather than an `ok=True` response without a `blob_ref`. A successful
            # PUT followed by a failed reply leaks one blob on M₁ — acceptable per
            # ADR-067 (server-side retention/TTL).
            try:
                image_bytes = saved_path.read_bytes()
                mime = f"image/{fmt}"
                store_ref = await self._blob_store.put(
                    image_bytes,
                    mime=mime,
                    source="imagecli",
                    filename=f"{request_id or 'unknown'}.{fmt}",
                )
                wire_blob_ref = WireBlobRef.from_store_ref(store_ref)
                if not wire_blob_ref.store_key:
                    raise ValueError(
                        "BlobStore.put returned an empty store_key; "
                        "a live ingest must yield a real store_key"
                    )

                # hoisted dict[str, Any] — inline **({...} if ...) makes pyright
                # type the spread as dict[str, str] and reject remaining kwargs
                job_kw: dict[str, Any] = {"job_id": job_id} if job_id is not None else {}
                resp = ImageResponse(
                    contract_version=CONTRACT_VERSION,
                    trace_id=trace_id,
                    issued_at=datetime.now(timezone.utc),
                    request_id=request_id,
                    ok=True,
                    blob_ref=wire_blob_ref,
                    mime_type=mime,
                    width=width,
                    height=height,
                    engine=engine_name,
                    seed_used=seed if seed is not None else 0,
                    **job_kw,
                )
                await self.reply(msg, resp.model_dump_json(exclude_none=True).encode())

            except Exception as e:
                # Sanitize before logging or wire-emitting — `str(e)` from httpx
                # exceptions can carry URLs and bearer tokens (#97 B2/W4).
                sanitized = _sanitize_delivery_exception(e)
                log.warning("Failed to deliver image via BlobStore: %s", sanitized)
                await self._reply_error(
                    msg,
                    trace_id,
                    request_id,
                    "delivery_failed",
                    sanitized,
                    job_id=job_id,
                )
            finally:
                if uses_per_request_cfg:
                    engine.cleanup()
                else:
                    # Registry-cached engine stays warm between requests.
                    engine.clear_cache()
                # Clean up tmp file unconditionally — BlobStore now owns persistence.
                try:
                    if saved_path.exists():
                        saved_path.unlink()
                except OSError:
                    pass

    async def _reply_error(
        self,
        msg: Any,
        trace_id: str,
        request_id: str,
        error: str,
        error_detail: str | None = None,
        job_id: str | None = None,
    ) -> None:
        """Send an error reply per the v0.4.x ImageResponse contract.

        Populates both the legacy free-text ``error: str`` field (for back-compat
        with consumers that read it directly) and the structured ``worker_error:
        WorkerError`` field (for v0.4.x consumers that route on canonical codes
        + retryability).
        """
        worker_err = _make_worker_error(error, error_detail)
        # ImageResponse.request_id is min_length=1; some failure paths reach here
        # before request_id is known (e.g. malformed envelope). Use
        # model_construct to skip validation in that single edge case — mirrors
        # the voiceCLI _err_tts pattern.
        safe_trace = trace_id or "unknown"
        now = datetime.now(timezone.utc)
        # hoisted dict[str, Any] — see success-path note on inline-spread pyright limits
        job_kw: dict[str, Any] = {"job_id": job_id} if job_id is not None else {}
        if request_id:
            resp = ImageResponse(
                contract_version=CONTRACT_VERSION,
                trace_id=safe_trace,
                issued_at=now,
                request_id=request_id,
                ok=False,
                error=error,
                worker_error=worker_err,
                **job_kw,
            )
        else:
            resp = ImageResponse.model_construct(
                contract_version=CONTRACT_VERSION,
                trace_id=safe_trace,
                issued_at=now,
                request_id="",
                ok=False,
                error=error,
                worker_error=worker_err,
                **job_kw,
            )
        await self.reply(msg, resp.model_dump_json(exclude_none=True).encode())

    def heartbeat_payload(self) -> dict:
        """Extend base heartbeat per factory.image contract (see roxabi_contracts.image.models.Heartbeat)."""
        base = super().heartbeat_payload()
        from imagecli.model_registry import model_registry

        loaded = model_registry.loaded_engines()
        # Contract exposes a single engine_loaded string; pick the MRU entry
        # when the registry caches several, else fall back to per-request tracking.
        base["engine_loaded"] = loaded[-1] if loaded else self._engine_loaded
        base["active_requests"] = self._active_request_count()

        free_mb = model_registry.vram_free_mb()
        total_mb = 0
        try:
            import torch

            if torch.cuda.is_available():
                total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except Exception:
            pass
        if total_mb:
            base["vram_used_mb"] = int(total_mb - free_mb)
            base["vram_total_mb"] = int(total_mb)
        return base

    def _active_request_count(self) -> int:
        """Return current active request count for heartbeat."""
        # Semaphore-based counting: unavailable permits = active requests
        # max_concurrent - _sem._value gives in-flight count
        return self.max_concurrent - self._sem._value
