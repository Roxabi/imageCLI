"""ImageNatsAdapter — imageCLI NATS satellite for async image generation requests."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from roxabi_nats import NatsAdapterBase

log = logging.getLogger(__name__)

SUBJECT = "lyra.image.generate.request"
HEARTBEAT_SUBJECT = "lyra.image.heartbeat"
SCHEMA_VERSION = "1"


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
            queue_group="image_workers",
            envelope_name="image_request",
            schema_version=SCHEMA_VERSION,
            heartbeat_subject=HEARTBEAT_SUBJECT,
            heartbeat_interval=heartbeat_interval,
            drain_timeout=drain_timeout,
        )
        self.default_engine = default_engine
        self.max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._engine_loaded: str | None = None

    def _validate_request(self, payload: dict) -> tuple[bool, str | None]:
        """Validate required fields. Returns (valid, error_message)."""
        if not payload.get("prompt"):
            return False, "missing_required_field: prompt"
        if not payload.get("engine"):
            return False, "missing_required_field: engine"
        return True, None

    async def handle(self, msg: Any, payload: dict) -> None:
        """Process an image generation request."""
        request_id = payload.get("request_id", "")

        # Validate required fields
        valid, error = self._validate_request(payload)
        if not valid:
            await self._reply_error(msg, request_id, "missing_required_field", error)
            return

        # TODO: Engine validation and generation logic (subsequent tasks)
        await self._reply_error(msg, request_id, "not_implemented", "handle not fully implemented")

    async def _reply_error(
        self, msg: Any, request_id: str, error: str, error_detail: str | None = None
    ) -> None:
        """Send an error reply following ADR-046 contract."""
        reply = {
            "contract_version": "1",
            "request_id": request_id,
            "ok": False,
            "error": error,
        }
        if error_detail:
            reply["error_detail"] = error_detail
        await self.reply(msg, json.dumps(reply).encode())

    def heartbeat_payload(self) -> dict:
        """Extend base heartbeat with engine_loaded and active_requests."""
        base = super().heartbeat_payload()
        base["engine_loaded"] = self._engine_loaded
        base["active_requests"] = self._active_request_count()
        return base

    def _active_request_count(self) -> int:
        """Return current active request count for heartbeat."""
        # Semaphore-based counting: unavailable permits = active requests
        # max_concurrent - _sem._value gives in-flight count
        return self.max_concurrent - self._sem._value
