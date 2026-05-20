"""ImageNatsAdapter — imageCLI NATS satellite for async image generation requests."""

from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roxabi_contracts.errors import WorkerError
from roxabi_contracts.image import ImageResponse
from roxabi_nats import NatsAdapterBase

from imagecli.paths import move_to_nats_output
from imagecli.nats.validators import _map_exception_to_error, _resolve_loras, _validate_request

log = logging.getLogger(__name__)

SUBJECT = "lyra.image.generate.request"
HEARTBEAT_SUBJECT = "lyra.image.heartbeat"
SCHEMA_VERSION = "1"

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

    Subscribes to `lyra.image.generate.request` and responds with generated images.
    Follows the satellite-bootstrap pattern established by voice adapters (ADR-039).
    """

    def __init__(
        self,
        default_engine: str = "flux2-klein",
        *,
        max_concurrent: int = 1,
        heartbeat_interval: float = 5.0,
        drain_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            subject=SUBJECT,
            queue_group="IMAGE_WORKERS",
            envelope_name="image",
            schema_version=SCHEMA_VERSION,
            heartbeat_subject=HEARTBEAT_SUBJECT,
            heartbeat_interval=heartbeat_interval,
            drain_timeout=drain_timeout,
            inbox_prefix="_inbox.imagecli-image",
            wait_ready=False,  # worker semantics — see NatsAdapterBase docstring
        )
        self.default_engine = default_engine
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

        # Acquire semaphore for concurrency control
        async with self._sem:
            # Validate required fields and bounds
            valid, error = _validate_request(payload)
            if not valid:
                await self._reply_error(
                    msg, trace_id, request_id, "missing_required_field", error
                )
                return

            engine_name = payload["engine"]
            prompt = payload["prompt"]

            # Import engine module lazily (heavy torch imports)
            try:
                from imagecli.engine import get_engine, preflight_check, list_engines
            except ImportError as e:
                await self._reply_error(
                    msg, trace_id, request_id, "engine_load_failed", f"Import error: {e}"
                )
                return

            # Validate engine name against registry
            engine_names = {e["name"] for e in list_engines()}
            if engine_name not in engine_names:
                await self._reply_error(msg, trace_id, request_id, "unknown_engine", engine_name)
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
                await self._reply_error(msg, trace_id, request_id, error_code, error_detail)
                return

            # Preflight check (VRAM/RAM validation)
            try:
                preflight_check(engine)
            except Exception as e:
                error_code, error_detail = _map_exception_to_error(e)
                await self._reply_error(msg, trace_id, request_id, error_code, error_detail)
                return

            # Extract generation params with defaults
            width = payload.get("width", 1024)
            height = payload.get("height", 1024)
            steps = payload.get("steps", 50)
            guidance = payload.get("guidance", 4.0)
            seed = payload.get("seed")
            negative_prompt = payload.get("negative_prompt", "")
            fmt = payload.get("format", "png")
            output_mode = payload.get("output_mode", "b64")

            # Generate image
            start_time = time.monotonic()
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
                await self._reply_error(msg, trace_id, request_id, error_code, error_detail)
                return

            # Read, encode, and deliver image.
            # `final_path` tracks the persistent file once the tmp has been moved;
            # `reply_sent` flips True only after the NATS reply is acknowledged so
            # that an encode/send failure after the move can clean up the orphan.
            final_path: Path | None = None
            reply_sent = False
            try:
                image_bytes = saved_path.read_bytes()

                deliver_as_file = output_mode == "file"
                image_b64: str | None = None
                file_path_str: str | None = None
                if not deliver_as_file:
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    # NATS server allows 50 MB but the contract caps base64
                    # replies at ~750 KB; larger payloads downgrade to file
                    # delivery on the shared FS / Syncthing-replicated dir.
                    if len(image_b64) > 750_000:
                        deliver_as_file = True
                        image_b64 = None

                if deliver_as_file:
                    try:
                        final_path = move_to_nats_output(saved_path, request_id, fmt)
                    except (OSError, shutil.Error, ValueError) as move_err:
                        # Generation succeeded; persistence to nats_out failed.
                        # Distinct from generation_failed so the hub can route it.
                        log.exception(f"Failed to persist generated image: {move_err}")
                        await self._reply_error(
                            msg,
                            trace_id,
                            request_id,
                            "delivery_failed",
                            "Failed to persist image",
                        )
                        return
                    file_path_str = str(final_path)

                resp = ImageResponse(
                    contract_version="1",
                    trace_id=trace_id,
                    issued_at=datetime.now(timezone.utc),
                    request_id=request_id,
                    ok=True,
                    image_b64=image_b64,
                    file_path=file_path_str,
                    mime_type=f"image/{fmt}",
                    width=width,
                    height=height,
                    engine=engine_name,
                    seed_used=seed if seed is not None else 0,
                )
                await self.reply(msg, resp.model_dump_json(exclude_none=True).encode())
                reply_sent = True

            except Exception as e:
                log.exception(f"Failed to encode response: {e}")
                await self._reply_error(
                    msg,
                    trace_id,
                    request_id,
                    "generation_failed",
                    "Failed to encode response",
                )
            finally:
                if uses_per_request_cfg:
                    engine.cleanup()
                else:
                    # Registry-cached engine stays warm between requests.
                    engine.clear_cache()
                # Clean up tmp file (a successful move makes saved_path.exists() false).
                try:
                    if saved_path.exists():
                        saved_path.unlink()
                except OSError:
                    pass
                # Clean up persistent file if the move succeeded but the reply
                # didn't — otherwise nats_out/ would accumulate orphans on every
                # encoder/transport failure after move.
                if final_path is not None and not reply_sent:
                    try:
                        final_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    async def _reply_error(
        self,
        msg: Any,
        trace_id: str,
        request_id: str,
        error: str,
        error_detail: str | None = None,
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
        if request_id:
            resp = ImageResponse(
                contract_version="1",
                trace_id=safe_trace,
                issued_at=now,
                request_id=request_id,
                ok=False,
                error=error,
                worker_error=worker_err,
            )
        else:
            resp = ImageResponse.model_construct(
                contract_version="1",
                trace_id=safe_trace,
                issued_at=now,
                request_id="",
                ok=False,
                error=error,
                worker_error=worker_err,
            )
        await self.reply(msg, resp.model_dump_json(exclude_none=True).encode())

    def heartbeat_payload(self) -> dict:
        """Extend base heartbeat per lyra.image contract (see roxabi_contracts.image.models.Heartbeat)."""
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
