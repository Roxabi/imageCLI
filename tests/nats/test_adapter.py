"""Tests for ImageNatsAdapter (issue #50).

Follows voiceCLI test_stt_adapter.py pattern:
- Lazy import shim for --collect-only
- MockMsg class with reply attribute
- Payload builders with sensible defaults
- Patch engine.generate at import site for coverage
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Lazy import shim — lets --collect-only succeed even before the module exists
# ---------------------------------------------------------------------------

try:
    from imagecli.nats.adapter import (
        ImageNatsAdapter,
    )

    _import_error = None
except ImportError as _e:
    _import_error = _e
    ImageNatsAdapter = None  # type: ignore[assignment,misc]


def test_adapter_binds_subjects_from_contracts() -> None:
    """ImageNatsAdapter must bind to subjects sourced from roxabi_contracts.image,
    not to hardcoded ``lyra.image.*`` string literals. Guards Cross-repo #1
    from #89: a subject rename upstream propagates via `uv sync` instead of
    requiring a manual sweep in this repo."""
    _require_imports()
    from roxabi_contracts.image import SUBJECTS

    assert ImageNatsAdapter is not None
    noop_store = MagicMock()
    noop_store.put = AsyncMock()
    adapter = ImageNatsAdapter(max_concurrent=1, blob_store=noop_store)  # type: ignore[arg-type]
    assert adapter.subject == SUBJECTS.image_request, (
        f"adapter.subject ({adapter.subject!r}) must equal SUBJECTS.image_request "
        f"({SUBJECTS.image_request!r}); legacy hardcoded literal still in use."
    )
    assert adapter._heartbeat_subject == SUBJECTS.image_heartbeat, (
        f"adapter._heartbeat_subject ({adapter._heartbeat_subject!r}) must equal "
        f"SUBJECTS.image_heartbeat ({SUBJECTS.image_heartbeat!r})."
    )
    assert adapter.queue_group == SUBJECTS.image_workers, (
        f"adapter.queue_group ({adapter.queue_group!r}) must equal "
        f"SUBJECTS.image_workers ({SUBJECTS.image_workers!r})."
    )


def _require_imports() -> None:
    if _import_error is not None:
        pytest.fail(f"imagecli.nats.adapter not yet implemented (RED): {_import_error}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockMsg:
    """Minimal NATS message stand-in: carries .reply + .data and records respond() calls."""

    def __init__(self, data: bytes = b"", reply_subject: str = "_INBOX.test") -> None:
        self.data = data
        self.reply = reply_subject
        self._published: list[bytes] = []

    async def respond(self, data: bytes) -> None:
        self._published.append(data)

    def last_reply(self) -> dict:
        assert self._published, "No reply published"
        return json.loads(self._published[-1])

    @property
    def responses(self) -> list[bytes]:
        return list(self._published)


def _valid_payload(
    *,
    request_id: str = "req-001",
    prompt: str = "a white cat on a red chair",
    engine: str = "flux2-klein",
    contract_version: str = "1",
) -> dict:
    payload: dict = {
        "contract_version": contract_version,
        "request_id": request_id,
        "prompt": prompt,
        "engine": engine,
    }
    return payload


def _make_adapter(**kwargs):  # type: ignore[return]
    assert ImageNatsAdapter is not None, "ImageNatsAdapter not imported"
    # Noop BlobStore — these tests exercise validation/error paths that
    # short-circuit before reaching put(). #97 success-path tests live in
    # `test_integration.py` with the `adapter_with_blob_store` fixture.
    noop_store = MagicMock()
    noop_store.put = AsyncMock()
    defaults: dict[str, object] = {
        "max_concurrent": 1,
        "blob_store": noop_store,
    }
    defaults.update(kwargs)
    adapter = ImageNatsAdapter(**defaults)  # type: ignore[arg-type]

    # Mock the reply method to capture replies on the msg object
    # This is needed because NatsAdapterBase.reply() uses a NATS connection
    async def _mock_reply(msg, data: bytes) -> None:
        await msg.respond(data)

    adapter.reply = _mock_reply  # type: ignore[method-assign]
    return adapter


def _assert_worker_error(
    reply: dict,
    *,
    code: str,
    retryable: bool,
    detail_contains: str | None = None,
) -> None:
    """Assert the reply carries a structured WorkerError with the expected shape.

    The legacy ``or {}`` fallback (used in tests prior to the WorkerError
    migration) passes vacuously when ``worker_error`` is absent; this helper
    requires the field to exist before drilling into ``code``/``retryable``/
    ``detail``.
    """
    assert reply["ok"] is False, f"expected ok=False, got {reply!r}"
    we = reply.get("worker_error")
    assert we is not None, f"worker_error missing from reply: {reply!r}"
    assert we.get("code") == code, f"expected code={code!r}, got {we.get('code')!r}"
    assert we.get("retryable") is retryable, (
        f"expected retryable={retryable}, got {we.get('retryable')!r}"
    )
    if detail_contains is not None:
        detail = we.get("detail") or ""
        assert detail_contains in detail, (
            f"expected detail containing {detail_contains!r}, got {detail!r}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImageNatsAdapter:
    """Test cases for ImageNatsAdapter validation and generation logic."""

    # ------------------------------------------------------------------
    # Case 1: missing prompt
    # ------------------------------------------------------------------
    def test_adapter_rejects_missing_prompt(self) -> None:
        """Missing prompt returns error."""
        _require_imports()
        # Arrange
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload()
        del payload["prompt"]

        # Act
        asyncio.run(adapter.handle(msg, payload))

        # Assert
        reply = msg.last_reply()
        assert reply["error"] == "missing_required_field"
        _assert_worker_error(
            reply, code="worker.validation", retryable=False, detail_contains="prompt"
        )

    # ------------------------------------------------------------------
    # Case 2: missing engine
    # ------------------------------------------------------------------
    def test_adapter_rejects_missing_engine(self) -> None:
        """Missing engine returns error."""
        _require_imports()
        # Arrange
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload()
        del payload["engine"]

        # Act
        asyncio.run(adapter.handle(msg, payload))

        # Assert
        reply = msg.last_reply()
        assert reply["error"] == "missing_required_field"
        _assert_worker_error(
            reply, code="worker.validation", retryable=False, detail_contains="engine"
        )

    # ------------------------------------------------------------------
    # Case 3: unknown engine
    # ------------------------------------------------------------------
    def test_adapter_rejects_unknown_engine(self) -> None:
        """Unknown engine name returns error."""
        _require_imports()
        # Arrange
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload(engine="nonexistent-engine")

        # Force get_engine onto the adapter module namespace (lazy import pattern)
        # This mirrors the voiceCLI approach where api is imported inside the handler
        import imagecli.nats.adapter as _mod
        import imagecli.engine as _engine

        _mod.get_engine = _engine.get_engine  # type: ignore[attr-defined]

        # Act
        asyncio.run(adapter.handle(msg, payload))

        # Assert - get_engine raises ValueError for unknown engines
        reply = msg.last_reply()
        assert reply["ok"] is False
        assert reply["error"] == "unknown_engine"

    # ------------------------------------------------------------------
    # Case 4: valid request returns success with image_b64
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Case 5: loras list payload threads through to engine constructor
    # ------------------------------------------------------------------
    def test_nats_accepts_loras_list_payload(self) -> None:
        """Payload with loras list → engine constructed with matching list[LoraSpec]."""
        _require_imports()
        from imagecli.lora_spec import LoraSpec

        # Arrange
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload(prompt="a cat", engine="flux2-klein")
        payload["loras"] = [
            {"path": "/.roxabi/imagecli/loras/style.safetensors", "trigger": "sty"},
            {"path": "/.roxabi/imagecli/loras/face.safetensors", "trigger": "fce"},
        ]

        captured_kwargs: dict = {}

        def _fake_get_engine(name, **kwargs):
            captured_kwargs.update(kwargs)
            mock_engine = MagicMock()
            mock_engine.cleanup = MagicMock()
            mock_engine.clear_cache = MagicMock()
            return mock_engine

        # Bypass path allowlist validation — this is a wiring test, not a security test
        with (
            patch("imagecli.nats.validators._validate_path", return_value=(True, None)),
            patch("imagecli.engine.get_engine", side_effect=_fake_get_engine),
            patch(
                "imagecli.engine.list_engines",
                return_value=[{"name": "flux2-klein"}],
            ),
            patch("imagecli.engine.preflight_check"),
        ):
            asyncio.run(adapter.handle(msg, payload))

        # Wiring assertion: loras= kwarg must be a 2-element list of LoraSpec
        assert "loras" in captured_kwargs, "get_engine was not called with loras= kwarg"
        loras = captured_kwargs["loras"]
        assert len(loras) == 2
        assert isinstance(loras[0], LoraSpec)
        assert loras[0].path == "/.roxabi/imagecli/loras/style.safetensors"
        assert loras[0].trigger == "sty"
        assert isinstance(loras[1], LoraSpec)
        assert loras[1].path == "/.roxabi/imagecli/loras/face.safetensors"
        assert loras[1].trigger == "fce"

    # ------------------------------------------------------------------
    # Case 6: legacy singular keys fold into 1-element loras list
    # ------------------------------------------------------------------
    def test_nats_legacy_singular_payload_still_works(self) -> None:
        """Legacy singular lora_path/lora_scale/trigger/embedding_path → 1-element loras list."""
        _require_imports()
        from imagecli.lora_spec import LoraSpec

        # Arrange
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload(prompt="a cat", engine="flux2-klein")
        payload["lora_path"] = "/.roxabi/imagecli/loras/myface.safetensors"
        payload["lora_scale"] = 1.2
        payload["trigger"] = "lyraface"
        payload["embedding_path"] = "/.roxabi/imagecli/embeddings/myface.safetensors"

        captured_kwargs: dict = {}

        def _fake_get_engine(name, **kwargs):
            captured_kwargs.update(kwargs)
            mock_engine = MagicMock()
            mock_engine.cleanup = MagicMock()
            mock_engine.clear_cache = MagicMock()
            return mock_engine

        # Bypass path allowlist validation — this is a wiring test, not a security test
        with (
            patch("imagecli.nats.validators._validate_path", return_value=(True, None)),
            patch("imagecli.engine.get_engine", side_effect=_fake_get_engine),
            patch(
                "imagecli.engine.list_engines",
                return_value=[{"name": "flux2-klein"}],
            ),
            patch("imagecli.engine.preflight_check"),
        ):
            asyncio.run(adapter.handle(msg, payload))

        # Regression guard: singular keys must fold into a 1-element loras list
        assert "loras" in captured_kwargs, "get_engine was not called with loras= kwarg"
        loras = captured_kwargs["loras"]
        assert len(loras) == 1
        assert isinstance(loras[0], LoraSpec)
        assert loras[0].path == "/.roxabi/imagecli/loras/myface.safetensors"
        assert loras[0].scale == 1.2
        assert loras[0].trigger == "lyraface"
        assert loras[0].embedding_path == "/.roxabi/imagecli/embeddings/myface.safetensors"

    # ------------------------------------------------------------------
    # Case 7: mixed form (loras list + singular key) → error response
    # ------------------------------------------------------------------
    def test_nats_rejects_mixed_form_payload(self) -> None:
        """Payload with both loras list and a singular key → error response (no crash)."""
        _require_imports()

        # Arrange
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload(prompt="a cat", engine="flux2-klein")
        payload["loras"] = [{"path": "/.roxabi/imagecli/loras/style.safetensors"}]
        payload["lora_path"] = "/.roxabi/imagecli/loras/face.safetensors"  # mixed — forbidden

        with patch(
            "imagecli.engine.list_engines",
            return_value=[{"name": "flux2-klein"}],
        ):
            asyncio.run(adapter.handle(msg, payload))

        reply = msg.last_reply()
        # Must surface as a validation error — not as generation_failed,
        # which would indicate the mixed-form payload slipped past
        # _validate_request and into the generation loop.
        assert reply["error"] in ("invalid_request", "missing_required_field")
        assert reply["error"] != "generation_failed"
        _assert_worker_error(reply, code="worker.validation", retryable=False)
        detail = ((reply.get("worker_error") or {}).get("detail") or "").lower()
        assert "loras" in detail or "mixed" in detail or "singular" in detail, (
            f"expected detail to mention loras/mixed/singular, got {detail!r}"
        )

    # Happy-path generation lives in tests/nats/test_integration.py
    # (`test_handle_success_returns_blob_ref` and the two delivery_failed
    # cases). Removed the legacy xfail `test_adapter_handles_valid_request`
    # that asserted `image_b64` in the reply — that surface no longer
    # exists post-#97 / ADR-067.

    # ------------------------------------------------------------------
    # Path-traversal sanitization (request_id + format)
    # ------------------------------------------------------------------
    def test_adapter_rejects_request_id_with_path_separator(self) -> None:
        """request_id with `/` is rejected before reaching the filesystem."""
        _require_imports()
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload(request_id="a/../../b")

        asyncio.run(adapter.handle(msg, payload))

        reply = msg.last_reply()
        assert reply["error"] == "missing_required_field"
        _assert_worker_error(
            reply, code="worker.validation", retryable=False, detail_contains="request_id"
        )

    def test_adapter_rejects_request_id_with_dots(self) -> None:
        _require_imports()
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload(request_id="..")

        asyncio.run(adapter.handle(msg, payload))

        reply = msg.last_reply()
        assert reply["ok"] is False
        assert reply["error"] == "missing_required_field"

    def test_adapter_rejects_unknown_format(self) -> None:
        _require_imports()
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload()
        payload["format"] = "png/../../etc"

        asyncio.run(adapter.handle(msg, payload))

        reply = msg.last_reply()
        assert reply["error"] == "missing_required_field"
        _assert_worker_error(
            reply, code="worker.validation", retryable=False, detail_contains="format"
        )

    # ------------------------------------------------------------------
    # trace_id propagation + fallback
    # ------------------------------------------------------------------
    def test_adapter_forwards_explicit_trace_id(self) -> None:
        """Inbound trace_id distinct from request_id is preserved on the reply."""
        _require_imports()
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        # Force an error path (missing prompt) to capture the reply envelope.
        payload = _valid_payload()
        payload["trace_id"] = "trace-abc-123"
        del payload["prompt"]

        asyncio.run(adapter.handle(msg, payload))

        reply = msg.last_reply()
        assert reply["trace_id"] == "trace-abc-123"
        assert reply["request_id"] == "req-001"

    def test_adapter_trace_id_falls_back_to_unknown(self) -> None:
        """When trace_id AND request_id are absent/empty, reply trace_id is 'unknown'.

        Exercises the ``model_construct`` branch in ``_reply_error`` — the only
        path reachable when request_id is empty (ImageResponse.request_id is
        min_length=1, so the validated constructor would refuse it).
        """
        _require_imports()
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        # _validate_request rejects empty request_id (REQUEST_ID_PATTERN), which
        # routes through _reply_error before request_id is replaced — but the
        # adapter's trace_id derivation runs first, so trace_id falls back to
        # "unknown" (payload has neither trace_id nor a usable request_id).
        payload = _valid_payload(request_id="")

        asyncio.run(adapter.handle(msg, payload))

        reply = msg.last_reply()
        assert reply["trace_id"] == "unknown"
        # The model_construct branch leaves request_id="" — the reply still
        # serializes (validation skipped) and carries the structured error.
        assert reply["ok"] is False
        we = reply.get("worker_error")
        assert we is not None
        assert we["code"] == "worker.validation"

    def test_adapter_accepts_allowed_formats(self) -> None:
        """`png`, `jpeg`, `webp` are the contract-allowed formats."""
        _require_imports()
        from imagecli.nats.validators import ALLOWED_FORMATS, _validate_request

        for fmt in ALLOWED_FORMATS:
            payload = _valid_payload()
            payload["format"] = fmt
            valid, err = _validate_request(payload)
            assert valid, f"format={fmt} unexpectedly rejected: {err}"

    # b64-overflow → file fallback regression test removed with #97 / ADR-067.
    # The b64/file dual-mode response surface was replaced by a single
    # `blob_ref` field; size-driven fallback no longer exists. Equivalent
    # delivery-failure coverage lives in `test_integration.py` via
    # `test_handle_blobstore_put_failure_returns_delivery_failed` and
    # `test_handle_reply_failure_after_put_returns_delivery_failed`.

    # ------------------------------------------------------------------
    # job_id echo — #1840
    # ------------------------------------------------------------------
    def test_job_id_echoed_on_error_path(self) -> None:
        """job_id from the request is echoed back on error replies (#1840)."""
        _require_imports()
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload()
        payload["job_id"] = "job-echo-err-001"
        del payload["prompt"]  # triggers missing_required_field on the error path

        asyncio.run(adapter.handle(msg, payload))

        reply = msg.last_reply()
        assert reply["ok"] is False
        assert reply.get("job_id") == "job-echo-err-001", (
            f"expected job_id='job-echo-err-001' echoed on error reply, got {reply.get('job_id')!r}"
        )

    def test_job_id_echoed_on_success_path(self) -> None:
        """job_id from the request is echoed back on the success reply (#1840).

        We patch ImageResponse to capture constructor kwargs (including job_id)
        without running full Pydantic validation — which would fail because our
        blob_ref is a MagicMock, not a real BlobRef. The captured kwargs are the
        ground-truth evidence that the adapter passed job_id through correctly.
        """
        _require_imports()
        import json
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_blob_store = MagicMock()
        mock_blob_store.put = AsyncMock(return_value=MagicMock())

        adapter = _make_adapter(max_concurrent=1, blob_store=mock_blob_store)
        msg = MockMsg()
        payload = _valid_payload()
        payload["job_id"] = "job-echo-ok-002"

        mock_engine = MagicMock()
        mock_engine.cleanup = MagicMock()
        mock_engine.clear_cache = MagicMock()

        fake_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(fake_image_bytes)
            fake_path = Path(tmp.name)

        mock_engine.generate = MagicMock(return_value=fake_path)

        # Intercept ImageResponse() — capture kwargs, return serializable mock.
        captured_kwargs: dict = {}

        def _fake_image_response(*args, **kwargs):  # type: ignore[return]
            captured_kwargs.update(kwargs)
            mock_resp = MagicMock()
            mock_resp.model_dump_json.return_value = json.dumps(
                {
                    "ok": True,
                    "job_id": kwargs.get("job_id", ""),
                    "request_id": kwargs.get("request_id", ""),
                }
            )
            return mock_resp

        # WireBlobRef: from_store_ref returns mock with truthy store_key.
        fake_wire_ref = MagicMock()
        fake_wire_ref.store_key = "blobstore://test/image.png"

        try:
            with (
                patch("imagecli.engine.get_engine", return_value=mock_engine),
                patch("imagecli.engine.list_engines", return_value=[{"name": "flux2-klein"}]),
                patch("imagecli.engine.preflight_check"),
                patch("imagecli.nats.adapter.WireBlobRef") as mock_wire_cls,
                patch("imagecli.nats.adapter.ImageResponse", side_effect=_fake_image_response),
                patch("imagecli.model_registry.model_registry") as mock_registry,
            ):
                mock_wire_cls.from_store_ref.return_value = fake_wire_ref
                mock_registry.get.return_value = mock_engine
                mock_registry.loaded_engines.return_value = []
                asyncio.run(adapter.handle(msg, payload))
        finally:
            try:
                os.unlink(fake_path)
            except OSError:
                pass

        # Primary assertion: adapter passed job_id= to ImageResponse constructor.
        assert captured_kwargs.get("job_id") == "job-echo-ok-002", (
            f"expected job_id='job-echo-ok-002' passed to ImageResponse, "
            f"got {captured_kwargs.get('job_id')!r} (full kwargs: {captured_kwargs})"
        )
        # Secondary: the serialized reply reflects job_id on the wire.
        reply = msg.last_reply()
        assert reply.get("job_id") == "job-echo-ok-002", (
            f"expected job_id='job-echo-ok-002' echoed on success reply, got {reply.get('job_id')!r}"
        )
