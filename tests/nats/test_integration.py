"""Integration tests for ImageNatsAdapter — end-to-end request/response cycle.

These tests mock the engine generation layer to avoid loading ML models,
but test the real adapter logic including validation, engine selection,
and response encoding.

Note: Tests requiring full generation flow are marked with xfail until
the handle() implementation is complete (ADR-046).
"""

from __future__ import annotations

import base64
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roxabi_nats import CONTRACT_VERSION

# Canned 1×1 PNG used across engine mocks and BlobRef assertions.
PNG_DATA = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
PNG_SHA = hashlib.sha256(PNG_DATA).hexdigest()


# ── Mock message class for NATS ───────────────────────────────────────────────


class MockNatsMessage:
    """Minimal mock of nats-py Msg for testing."""

    def __init__(self, data: bytes, reply_subject: str = "reply.subject"):
        self.data = data
        self.reply = reply_subject
        self._published: list[bytes] = []

    async def respond(self, data: bytes) -> None:
        """Capture response data."""
        self._published.append(data)

    def last_reply(self) -> dict:
        """Get the last response as parsed JSON."""
        assert self._published, "No reply published"
        return json.loads(self._published[-1])


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_engine():
    """Create a mock engine that writes a fake image file."""
    engine = MagicMock()
    engine.name = "flux2-klein"
    engine.vram_gb = 8.0

    def mock_generate(prompt, *, output_path, **kwargs):
        # Write a minimal valid PNG to the output path
        output_path.write_bytes(PNG_DATA)
        # Adapter expects engine.generate to return the saved path so it can
        # read the bytes back via saved_path.read_bytes() (#97 handle() flow).
        return output_path

    engine.generate.side_effect = mock_generate
    engine.cleanup = MagicMock()
    return engine


@pytest.fixture
def adapter():
    """Create an ImageNatsAdapter instance for testing.

    Passes a noop BlobStore mock so legacy validation-path tests (which never
    reach put()) don't need their own per-test wiring. #97 success-path tests
    use `adapter_with_blob_store` to wire a canned BlobRef.
    """
    from imagecli.nats.adapter import ImageNatsAdapter

    noop_blob_store = MagicMock()
    noop_blob_store.put = AsyncMock()
    adapter = ImageNatsAdapter(default_engine="flux2-klein", blob_store=noop_blob_store)

    async def _mock_reply(msg, data: bytes) -> None:
        await msg.respond(data)

    adapter.reply = _mock_reply  # type: ignore[method-assign]
    return adapter


@pytest.fixture
def mock_nc():
    """Create a mock NATS client."""
    nc = MagicMock()
    nc.is_connected = True
    nc.publish = AsyncMock()
    return nc


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_adapter_handles_missing_prompt(adapter, mock_nc):
    """Adapter returns error when prompt is missing."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "request_id": "test-req-missing-prompt",
        "engine": "flux2-klein",
        # No prompt field
    }
    msg = MockNatsMessage(b"test")

    # Act
    await adapter.handle(msg, request_payload)

    # Assert: error response
    response = msg.last_reply()
    assert response["error"] == "missing_required_field"
    we = response.get("worker_error")
    assert we is not None
    assert we["code"] == "worker.validation"
    assert we["retryable"] is False
    assert "prompt" in (we.get("detail") or "")
    assert response["request_id"] == "test-req-missing-prompt"


@pytest.mark.asyncio
async def test_adapter_handles_missing_engine(adapter, mock_nc):
    """Adapter returns error when engine is missing."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "request_id": "test-req-missing-engine",
        "prompt": "test prompt",
        # No engine field
    }
    msg = MockNatsMessage(b"test")

    # Act
    await adapter.handle(msg, request_payload)

    # Assert: error response
    response = msg.last_reply()
    assert response["error"] == "missing_required_field"
    we = response.get("worker_error")
    assert we is not None
    assert we["code"] == "worker.validation"
    assert we["retryable"] is False
    assert "engine" in (we.get("detail") or "")


@pytest.mark.asyncio
async def test_adapter_handles_unknown_engine(adapter, mock_nc):
    """Adapter returns error when engine name is unknown."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "request_id": "test-req-bad-engine",
        "prompt": "test prompt",
        "engine": "nonexistent-engine",
    }
    msg = MockNatsMessage(b"test")

    with patch("imagecli.engine.get_engine", side_effect=ValueError("unknown engine")):
        # Act
        await adapter.handle(msg, request_payload)

    # Assert: error response
    response = msg.last_reply()
    assert response["ok"] is False
    assert "unknown_engine" in response["error"]


@pytest.mark.asyncio
async def test_adapter_handles_preflight_failure(adapter, mock_engine, mock_nc):
    """Adapter returns error when preflight check fails (insufficient VRAM)."""
    # Arrange
    from imagecli.engine import InsufficientResourcesError

    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "request_id": "test-req-preflight-fail",
        "prompt": "test prompt",
        "engine": "flux2-klein",
    }
    msg = MockNatsMessage(b"test")

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine),
        patch(
            "imagecli.engine.preflight_check", side_effect=InsufficientResourcesError("low VRAM")
        ),
    ):
        # Act
        await adapter.handle(msg, request_payload)

    # Assert: error response
    response = msg.last_reply()
    assert response["ok"] is False
    assert "insufficient_resources" in response["error"]


@pytest.mark.asyncio
async def test_adapter_handles_generation_failure(adapter, mock_engine, mock_nc):
    """Adapter returns error when generation fails."""
    # Arrange
    mock_engine.generate.side_effect = RuntimeError("generation failed")

    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "request_id": "test-req-gen-fail",
        "prompt": "test prompt",
        "engine": "flux2-klein",
    }
    msg = MockNatsMessage(b"test")

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine),
        patch("imagecli.engine.preflight_check"),
    ):
        # Act
        await adapter.handle(msg, request_payload)

    # Assert: error response
    response = msg.last_reply()
    assert response["ok"] is False
    assert "generation_failed" in response["error"]


@pytest.mark.asyncio
async def test_adapter_uses_default_engine(adapter, mock_engine, mock_nc, tmp_path):
    """Adapter uses default engine when not specified in request."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "request_id": "test-req-default-engine",
        "prompt": "test prompt",
        # No engine specified - should use default
    }
    msg = MockNatsMessage(b"test")

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine) as mock_get_engine,
        patch("imagecli.engine.preflight_check"),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = str(tmp_path / "test_image.png")
        mock_tmp_file.__enter__ = MagicMock(return_value=mock_tmp_file)
        mock_tmp_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_tmp_file

        (tmp_path / "test_image.png").write_bytes(PNG_DATA)

        # Act
        await adapter.handle(msg, request_payload)

    # Assert: default engine was used
    mock_get_engine.assert_called_once()
    engine_arg = mock_get_engine.call_args.args[0]
    assert engine_arg == "flux2-klein"  # default from fixture

    # Assert: success response
    response = msg.last_reply()
    assert response["ok"] is True


@pytest.mark.asyncio
async def test_adapter_handles_lora_params(adapter, mock_engine, mock_nc, tmp_path):
    """Adapter passes LoRA parameters to engine lookup."""
    # Arrange
    request_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "request_id": "test-req-lora",
        "prompt": "test prompt",
        "engine": "flux2-klein",
        "lora_path": "/path/to/lora.safetensors",
        "lora_scale": 1.5,
        "trigger": "lyraface",
        "embedding_path": "/path/to/emb.safetensors",
    }
    msg = MockNatsMessage(b"test")

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine) as mock_get_engine,
        patch("imagecli.engine.preflight_check"),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        mock_tmp_file = MagicMock()
        mock_tmp_file.name = str(tmp_path / "test_image.png")
        mock_tmp_file.__enter__ = MagicMock(return_value=mock_tmp_file)
        mock_tmp_file.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value = mock_tmp_file

        (tmp_path / "test_image.png").write_bytes(PNG_DATA)

        # Act
        await adapter.handle(msg, request_payload)

    # Assert: LoRA params passed to get_engine
    mock_get_engine.assert_called_once_with(
        "flux2-klein",
        lora_path="/path/to/lora.safetensors",
        lora_scale=1.5,
        trigger="lyraface",
        embedding_path="/path/to/emb.safetensors",
    )

    # Assert: success response
    response = msg.last_reply()
    assert response["ok"] is True


# ── BlobRef migration tests (#97 slice 2 RED) ─────────────────────────────────


@pytest.fixture
def mock_blob_store():
    """Mock BlobStore returning a canned BlobRef on put() (ADR-067)."""
    from datetime import UTC, datetime

    from roxabi_blobs.models import BlobRef

    store = MagicMock()
    canned_ref = BlobRef(
        store_key="ck-test",
        content_hash=PNG_SHA,
        mime="image/png",
        size=len(PNG_DATA),
        source="imagecli",
        created_at=datetime.now(tz=UTC),
    )
    store.put = AsyncMock(return_value=canned_ref)
    return store


@pytest.fixture
def adapter_with_blob_store(mock_blob_store):
    """Adapter with mocked BlobStore wired (slice 2 — #97 T12)."""
    from imagecli.nats.adapter import ImageNatsAdapter

    adapter = ImageNatsAdapter(default_engine="flux2-klein", blob_store=mock_blob_store)

    async def _mock_reply(msg, data: bytes) -> None:
        await msg.respond(data)

    adapter.reply = _mock_reply  # type: ignore[method-assign]
    return adapter


def _success_payload(request_id: str = "test-blobref-success") -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "schema_version": 1,
        "request_id": request_id,
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


def _patch_engine_layer(tmp_path, mock_engine):
    """Standard engine-layer mocks for adapter integration tests."""
    mock_tmp_file = MagicMock()
    mock_tmp_file.name = str(tmp_path / "test_image.png")
    mock_tmp_file.__enter__ = MagicMock(return_value=mock_tmp_file)
    mock_tmp_file.__exit__ = MagicMock(return_value=False)
    (tmp_path / "test_image.png").write_bytes(PNG_DATA)
    return mock_tmp_file


@pytest.mark.asyncio
async def test_handle_success_returns_blob_ref(
    adapter_with_blob_store, mock_blob_store, mock_engine, tmp_path
):
    """Happy path: handle() PUTs bytes to BlobStore and emits ImageResponse with blob_ref."""
    payload = _success_payload()
    msg = MockNatsMessage(json.dumps(payload).encode())
    mock_tmp_file = _patch_engine_layer(tmp_path, mock_engine)

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine),
        patch("imagecli.engine.preflight_check"),
        patch("imagecli.model_registry.model_registry.get", return_value=mock_engine),
        patch("tempfile.NamedTemporaryFile", return_value=mock_tmp_file),
    ):
        await adapter_with_blob_store.handle(msg, payload)

    mock_blob_store.put.assert_awaited_once()
    put_kwargs = mock_blob_store.put.call_args.kwargs
    assert put_kwargs.get("mime") == "image/png"
    assert put_kwargs.get("source") == "imagecli"
    assert put_kwargs.get("filename") == "test-blobref-success.png"

    response = msg.last_reply()
    assert response["ok"] is True
    assert response["request_id"] == payload["request_id"]
    assert "blob_ref" in response
    blob_ref = response["blob_ref"]
    assert blob_ref["store_key"] == "ck-test"
    assert blob_ref["content_hash"] == PNG_SHA
    assert blob_ref["mime"] == "image/png"
    assert blob_ref["size"] == len(PNG_DATA)
    assert "image_b64" not in response
    assert "file_path" not in response


@pytest.mark.asyncio
async def test_handle_blobstore_put_failure_returns_delivery_failed(
    adapter_with_blob_store, mock_blob_store, mock_engine, tmp_path
):
    """HttpBlobStore.put() failure routes to delivery_failed / worker.internal / retryable=True."""
    import httpx

    mock_blob_store.put = AsyncMock(side_effect=httpx.HTTPError("simulated put failure"))

    payload = _success_payload(request_id="test-blobref-put-fail")
    msg = MockNatsMessage(json.dumps(payload).encode())
    mock_tmp_file = _patch_engine_layer(tmp_path, mock_engine)

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine),
        patch("imagecli.engine.preflight_check"),
        patch("imagecli.model_registry.model_registry.get", return_value=mock_engine),
        patch("tempfile.NamedTemporaryFile", return_value=mock_tmp_file),
    ):
        await adapter_with_blob_store.handle(msg, payload)

    response = msg.last_reply()
    assert response["ok"] is False
    assert response["error"] == "delivery_failed"
    we = response.get("worker_error")
    assert we is not None
    assert we["code"] == "worker.internal"
    assert we["retryable"] is True
    # finally block must unlink the tmp file even when put() raises
    assert not (tmp_path / "test_image.png").exists()


@pytest.mark.asyncio
async def test_handle_reply_failure_after_put_returns_delivery_failed(
    adapter_with_blob_store, mock_blob_store, mock_engine, tmp_path
):
    """Post-PUT / pre-reply failure: PUT succeeds, reply() raises → still emits delivery_failed.

    Single-try wrap (spec N2): the response build + reply call live in the same try
    block as the put() call so any post-PUT exception routes through _reply_error
    rather than leaking an ok=True response without a blob_ref.
    """
    payload = _success_payload(request_id="test-blobref-reply-fail")
    msg = MockNatsMessage(json.dumps(payload).encode())
    mock_tmp_file = _patch_engine_layer(tmp_path, mock_engine)

    # First reply (happy path build) raises; _reply_error's reply must still land.
    reply_calls: list[bytes] = []
    raise_once = {"done": False}

    async def _flaky_reply(target_msg, data: bytes) -> None:
        if not raise_once["done"]:
            raise_once["done"] = True
            raise RuntimeError("simulated reply transport failure")
        reply_calls.append(data)
        await target_msg.respond(data)

    adapter_with_blob_store.reply = _flaky_reply  # type: ignore[method-assign]

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine),
        patch("imagecli.engine.preflight_check"),
        patch("imagecli.model_registry.model_registry.get", return_value=mock_engine),
        patch("tempfile.NamedTemporaryFile", return_value=mock_tmp_file),
    ):
        await adapter_with_blob_store.handle(msg, payload)

    mock_blob_store.put.assert_awaited_once()
    response = msg.last_reply()
    assert response["ok"] is False
    assert response["error"] == "delivery_failed"
    we = response.get("worker_error")
    assert we is not None
    assert we["code"] == "worker.internal"
    assert we["retryable"] is True
