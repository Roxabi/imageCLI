"""Tests for ImageNatsAdapter (issue #50).

Follows voiceCLI test_stt_adapter.py pattern:
- Lazy import shim for --collect-only
- MockMsg class with reply attribute
- Payload builders with sensible defaults
- Patch engine.generate at import site for coverage
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _valid_image_b64() -> str:
    """Return a minimal valid base64-encoded PNG-ish byte blob."""
    # Minimal PNG: 8-byte header + minimal IHDR + IDAT + IEND
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()


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
    defaults: dict[str, object] = {
        "max_concurrent": 1,
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
            {"path": "/ComfyUI/models/loras/style.safetensors", "trigger": "sty"},
            {"path": "/ComfyUI/models/loras/face.safetensors", "trigger": "fce"},
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
        assert loras[0].path == "/ComfyUI/models/loras/style.safetensors"
        assert loras[0].trigger == "sty"
        assert isinstance(loras[1], LoraSpec)
        assert loras[1].path == "/ComfyUI/models/loras/face.safetensors"
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
        payload["lora_path"] = "/ComfyUI/models/loras/myface.safetensors"
        payload["lora_scale"] = 1.2
        payload["trigger"] = "lyraface"
        payload["embedding_path"] = "/ComfyUI/models/embeddings/myface.safetensors"

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
        assert loras[0].path == "/ComfyUI/models/loras/myface.safetensors"
        assert loras[0].scale == 1.2
        assert loras[0].trigger == "lyraface"
        assert loras[0].embedding_path == "/ComfyUI/models/embeddings/myface.safetensors"

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
        payload["loras"] = [{"path": "/ComfyUI/models/loras/style.safetensors"}]
        payload["lora_path"] = "/ComfyUI/models/loras/face.safetensors"  # mixed — forbidden

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

    @pytest.mark.xfail(reason="Adapter needs generation logic in handle() - ADR-046")
    def test_adapter_handles_valid_request(self, tmp_path: Path) -> None:
        """Valid request returns success with image_b64."""
        _require_imports()
        # Arrange
        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload(prompt="a white cat", engine="flux2-klein")

        # Create a mock engine with generate method that writes to temp path
        mock_engine = MagicMock()
        mock_image_path = tmp_path / "output.png"
        mock_image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        mock_engine.generate.return_value = mock_image_path

        # Force get_engine onto the adapter module namespace (lazy import pattern)
        import imagecli.nats.adapter as _mod
        import imagecli.engine as _engine

        _mod.get_engine = _engine.get_engine  # type: ignore[attr-defined]

        with patch("imagecli.nats.adapter.get_engine", return_value=mock_engine):
            # Act
            asyncio.run(adapter.handle(msg, payload))

        # Assert
        reply = msg.last_reply()
        assert reply["ok"] is True
        assert reply["contract_version"] == "1"
        assert reply["request_id"] == "req-001"
        assert "image_b64" in reply
        assert reply["image_b64"] == _valid_image_b64()
        mock_engine.generate.assert_called_once()

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

    # ------------------------------------------------------------------
    # b64-overflow → file fallback (regression guard for the
    # filename-vs-absolute-path bug fixed in a620f4b)
    # ------------------------------------------------------------------
    def test_b64_overflow_falls_back_to_absolute_file_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _require_imports()
        # Redirect nats_out into tmp_path so we don't touch ~/.roxabi/
        monkeypatch.setenv("IMAGECLI_NATS_OUTPUT_DIR", str(tmp_path / "nats_out"))

        adapter = _make_adapter(max_concurrent=1)
        msg = MockMsg()
        payload = _valid_payload(request_id="abcdef12")

        # Mock engine that writes a >562 KB file (≈ 750 KB once base64-encoded).
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (700 * 1024)
        mock_engine = MagicMock()
        mock_engine.name = "flux2-klein"
        mock_engine.vram_gb = 8.0

        def _gen(_prompt, *, output_path, **_kw):
            output_path.write_bytes(big)
            return output_path

        mock_engine.generate.side_effect = _gen
        mock_engine.cleanup = MagicMock()
        mock_engine.clear_cache = MagicMock()

        # Stub out preflight + registry lookup; force our mock engine into the path.
        with (
            patch("imagecli.engine.preflight_check"),
            patch("imagecli.model_registry.model_registry.get", return_value=mock_engine),
        ):
            asyncio.run(adapter.handle(msg, payload))

        reply = msg.last_reply()
        assert reply["ok"] is True, f"expected success, got {reply}"
        # Overflow path: file_path present, image_b64 absent
        assert "file_path" in reply
        assert "image_b64" not in reply
        # Regression guard: must be an absolute path, not a bare filename
        fp = Path(reply["file_path"])
        assert fp.is_absolute()
        assert fp.parent == (tmp_path / "nats_out").resolve()
        assert fp.name == "nats_abcdef12.png"
        # Registry-cached engines stay warm between requests; the per-request
        # contract is `clear_cache()` in `finally` (¬`cleanup()`, which would
        # tear the engine down). Lock that down.
        assert mock_engine.clear_cache.called, "clear_cache must run on the registry-warm path"
        assert not mock_engine.cleanup.called, "cleanup must NOT run on the registry-warm path"
        assert fp.exists()
