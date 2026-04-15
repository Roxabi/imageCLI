"""Integration tests for ImageNatsAdapter — end-to-end request/response cycle.

These tests mock the engine generation layer to avoid loading ML models,
but test the real adapter logic including validation, engine selection,
and response encoding.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxabi_nats import CONTRACT_VERSION


# ── Mock message class for NATS ───────────────────────────────────────────────


class MockNatsMessage:
    """Minimal mock of nats-py Msg for testing."""

    def __init__(self, data: bytes, reply_subject: str = "reply.subject"):
        self.data = data
        self.reply = reply_subject


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_engine():
    """Create a mock engine that writes a fake image file."""
    engine = MagicMock()
    engine.name = "flux2-klein"
    engine.vram_gb = 8.0

    def mock_generate(prompt, *, output_path, **kwargs):
        # Write a minimal valid PNG to the output path
        # 1x1 red pixel PNG
        png_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        output_path.write_bytes(png_data)

    engine.generate.side_effect = mock_generate
    engine.cleanup = MagicMock()
    return engine


@pytest.fixture
def adapter():
    """Create an ImageNatsAdapter instance for testing."""
    from imagecli.nats.adapter import ImageNatsAdapter

    return ImageNatsAdapter(default_engine="flux2-klein")


@pytest.fixture
def mock_nc():
    """Create a mock NATS client."""
    nc = MagicMock()
    nc.is_connected = True
    nc.publish = AsyncMock()
    return nc


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_adapter_handles_request_end_to_end(adapter, mock_engine, mock_nc, tmp_path):
    """Full request/response cycle: validate -> get engine -> generate -> reply."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": "1",
        "request_id": "test-req-123",
        "prompt": "a white cat on a red chair",
        "engine": "flux2-klein",
        "width": 512,
        "height": 512,
        "steps": 20,
        "guidance": 4.0,
        "seed": 42,
        "negative_prompt": "",
        "format": "png",
    }
    msg = MockNatsMessage(json.dumps(request_payload).encode())

    # Track the reply
    reply_data = None

    async def capture_reply(subject, data):
        nonlocal reply_data
        reply_data = data

    mock_nc.publish.side_effect = capture_reply
    adapter._nc = mock_nc

    # Mock the engine layer - imports are done inside handle(), so patch at source
    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine) as mock_get_engine,
        patch("imagecli.engine.preflight_check") as mock_preflight,
        patch("imagecli.nats.adapter.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        # Setup temp file to use tmp_path
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = str(tmp_path / "test_image.png")
        mock_tmp_file.__enter__ = MagicMock(return_value=mock_tmp_file)
        mock_tmp_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_tmp_file

        # Ensure the temp file exists for read_bytes
        (tmp_path / "test_image.png").write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
            )
        )

        # Act
        await adapter.handle(msg, request_payload)

    # Assert: engine was obtained with correct params
    mock_get_engine.assert_called_once_with(
        "flux2-klein",
        lora_path=None,
        lora_scale=1.0,
        trigger=None,
        embedding_path=None,
    )

    # Assert: preflight was called
    mock_preflight.assert_called_once_with(mock_engine)

    # Assert: engine.generate was called
    mock_engine.generate.assert_called_once()
    call_args = mock_engine.generate.call_args
    # prompt is positional, other params are kwargs
    assert call_args.args[0] == "a white cat on a red chair"
    assert call_args.kwargs["width"] == 512
    assert call_args.kwargs["height"] == 512
    assert call_args.kwargs["steps"] == 20
    assert call_args.kwargs["guidance"] == 4.0
    assert call_args.kwargs["seed"] == 42
    assert call_args.kwargs["negative_prompt"] == ""

    # Assert: reply was sent
    assert reply_data is not None
    response = json.loads(reply_data)
    assert response["contract_version"] == CONTRACT_VERSION
    assert response["request_id"] == "test-req-123"
    assert response["ok"] is True
    assert "image_b64" in response
    assert response["format"] == "png"
    assert response["width"] == 512
    assert response["height"] == 512
    assert "duration_s" in response


@pytest.mark.asyncio
async def test_adapter_handles_missing_prompt(adapter, mock_nc):
    """Adapter returns error when prompt is missing."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": "1",
        "request_id": "test-req-missing-prompt",
        # No prompt field
    }

    # Track the reply
    reply_data = None

    async def capture_reply(subject, data):
        nonlocal reply_data
        reply_data = data

    mock_nc.publish.side_effect = capture_reply
    adapter._nc = mock_nc

    # Act
    await adapter.handle(MockNatsMessage(b"test"), request_payload)

    # Assert: error response
    assert reply_data is not None
    response = json.loads(reply_data)
    assert response["ok"] is False
    assert "missing_required_field: prompt" in response["error"]
    assert response["request_id"] == "test-req-missing-prompt"


@pytest.mark.asyncio
async def test_adapter_handles_unknown_engine(adapter, mock_nc):
    """Adapter returns error when engine name is unknown."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": "1",
        "request_id": "test-req-bad-engine",
        "prompt": "test prompt",
        "engine": "nonexistent-engine",
    }

    # Track the reply
    reply_data = None

    async def capture_reply(subject, data):
        nonlocal reply_data
        reply_data = data

    mock_nc.publish.side_effect = capture_reply
    adapter._nc = mock_nc

    with patch("imagecli.engine.get_engine", side_effect=ValueError("unknown engine")):
        # Act
        await adapter.handle(MockNatsMessage(b"test"), request_payload)

    # Assert: error response
    assert reply_data is not None
    response = json.loads(reply_data)
    assert response["ok"] is False
    assert "unknown_engine" in response["error"]


@pytest.mark.asyncio
async def test_adapter_handles_preflight_failure(adapter, mock_engine, mock_nc):
    """Adapter returns error when preflight check fails (insufficient VRAM)."""
    # Arrange
    from imagecli.engine import InsufficientResourcesError

    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": "1",
        "request_id": "test-req-preflight-fail",
        "prompt": "test prompt",
        "engine": "flux2-klein",
    }

    # Track the reply
    reply_data = None

    async def capture_reply(subject, data):
        nonlocal reply_data
        reply_data = data

    mock_nc.publish.side_effect = capture_reply
    adapter._nc = mock_nc

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine),
        patch("imagecli.engine.preflight_check", side_effect=InsufficientResourcesError("low VRAM")),
    ):
        # Act
        await adapter.handle(MockNatsMessage(b"test"), request_payload)

    # Assert: error response
    assert reply_data is not None
    response = json.loads(reply_data)
    assert response["ok"] is False
    assert "insufficient_resources" in response["error"]


@pytest.mark.asyncio
async def test_adapter_handles_generation_failure(adapter, mock_engine, mock_nc):
    """Adapter returns error when generation fails."""
    # Arrange
    mock_engine.generate.side_effect = RuntimeError("generation failed")

    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": "1",
        "request_id": "test-req-gen-fail",
        "prompt": "test prompt",
        "engine": "flux2-klein",
    }

    # Track the reply
    reply_data = None

    async def capture_reply(subject, data):
        nonlocal reply_data
        reply_data = data

    mock_nc.publish.side_effect = capture_reply
    adapter._nc = mock_nc

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine),
        patch("imagecli.engine.preflight_check"),
    ):
        # Act
        await adapter.handle(MockNatsMessage(b"test"), request_payload)

    # Assert: error response
    assert reply_data is not None
    response = json.loads(reply_data)
    assert response["ok"] is False
    assert "generation_failed" in response["error"]


@pytest.mark.asyncio
async def test_adapter_uses_default_engine(adapter, mock_engine, mock_nc, tmp_path):
    """Adapter uses default engine when not specified in request."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": "1",
        "request_id": "test-req-default-engine",
        "prompt": "test prompt",
        # No engine specified
    }

    # Track the reply
    reply_data = None

    async def capture_reply(subject, data):
        nonlocal reply_data
        reply_data = data

    mock_nc.publish.side_effect = capture_reply
    adapter._nc = mock_nc

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine) as mock_get_engine,
        patch("imagecli.engine.preflight_check"),
        patch("imagecli.nats.adapter.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = str(tmp_path / "test_image.png")
        mock_tmp_file.__enter__ = MagicMock(return_value=mock_tmp_file)
        mock_tmp_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_tmp_file

        (tmp_path / "test_image.png").write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
            )
        )

        # Act
        await adapter.handle(MockNatsMessage(b"test"), request_payload)

    # Assert: default engine was used
    mock_get_engine.assert_called_once()
    engine_arg = mock_get_engine.call_args.args[0]
    assert engine_arg == "flux2-klein"  # default from fixture

    # Assert: success response
    response = json.loads(reply_data)
    assert response["ok"] is True


@pytest.mark.asyncio
async def test_adapter_handles_lora_params(adapter, mock_engine, mock_nc, tmp_path):
    """Adapter passes LoRA parameters to engine lookup."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": "1",
        "request_id": "test-req-lora",
        "prompt": "test prompt",
        "engine": "flux2-klein",
        "lora_path": "/path/to/lora.safetensors",
        "lora_scale": 1.5,
        "trigger": "lyraface",
        "embedding_path": "/path/to/emb.safetensors",
    }

    # Track the reply
    reply_data = None

    async def capture_reply(subject, data):
        nonlocal reply_data
        reply_data = data

    mock_nc.publish.side_effect = capture_reply
    adapter._nc = mock_nc

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine) as mock_get_engine,
        patch("imagecli.engine.preflight_check"),
        patch("imagecli.nats.adapter.tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = str(tmp_path / "test_image.png")
        mock_tmp_file.__enter__ = MagicMock(return_value=mock_tmp_file)
        mock_tmp_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_tmp_file

        (tmp_path / "test_image.png").write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
            )
        )

        # Act
        await adapter.handle(MockNatsMessage(b"test"), request_payload)

    # Assert: LoRA params passed to get_engine
    mock_get_engine.assert_called_once_with(
        "flux2-klein",
        lora_path="/path/to/lora.safetensors",
        lora_scale=1.5,
        trigger="lyraface",
        embedding_path="/path/to/emb.safetensors",
    )

    # Assert: success response
    response = json.loads(reply_data)
    assert response["ok"] is True
