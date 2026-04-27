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
        assert reply["ok"] is False
        assert reply["error"] == "missing_required_field"
        assert "prompt" in reply.get("error_detail", "")

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
        assert reply["ok"] is False
        assert reply["error"] == "missing_required_field"
        assert "engine" in reply.get("error_detail", "")

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
        assert reply["ok"] is False
        # Must surface as a validation error — not as generation_failed,
        # which would indicate the mixed-form payload slipped past
        # _validate_request and into the generation loop.
        assert reply["error"] in ("invalid_request", "missing_required_field")
        assert reply["error"] != "generation_failed"
        assert (
            "loras" in reply.get("error_detail", "").lower()
            or "mixed" in reply.get("error_detail", "").lower()
            or "singular" in reply.get("error_detail", "").lower()
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
