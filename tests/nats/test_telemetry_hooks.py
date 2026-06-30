"""OTel lifecycle hook tests for imageCLI NATS adapter (#2069 Block 7)."""

from __future__ import annotations

import pytest
from roxabi_otel import InMemorySpanRecorder

from roxabi_contracts.telemetry import ATTR_BLOB_REF_OUT, ATTR_ENGINE
from imagecli.nats.adapter import ImageNatsAdapter

_TRACE = "550e8400-e29b-41d4-a716-446655440000"
_JOB = "a" * 32


class TestImageTelemetryHooks:
    @pytest.mark.asyncio
    async def test_adapter_records_engine_and_blob_out(self) -> None:
        recorder = InMemorySpanRecorder()
        adapter = ImageNatsAdapter(
            blob_store=type("BS", (), {"put": None})(),  # type: ignore[arg-type]
            lifecycle_hooks=recorder.hooks("imagecli-gen"),
        )
        adapter._otel_work_attrs = {ATTR_BLOB_REF_OUT: "image/out/key"}
        msg = type("M", (), {"subject": "factory.image.generate.request"})()
        payload = {
            "trace_id": _TRACE,
            "job_id": _JOB,
            "request_id": "req-1",
            "engine": "flux2-klein",
            "prompt": "cat",
        }

        async def _noop_handle(_msg: object, _payload: dict) -> None:
            return None

        adapter.handle = _noop_handle  # type: ignore[method-assign]
        await adapter._invoke_handle_with_hooks(msg, payload)

        spans = recorder.finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[ATTR_BLOB_REF_OUT] == "image/out/key"
        assert spans[0].attributes[ATTR_ENGINE] == "flux2-klein"
        assert "prompt" not in spans[0].attributes