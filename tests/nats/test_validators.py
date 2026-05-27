"""Tests for nats.validators._map_exception_to_error dispatch.

Guards the regression risk identified in audit Finding 5: if the registry
raise-site message changes, a substring-based dispatch silently
mis-routes unknown-engine errors to ``generation_failed`` (retryable),
when it should remain ``unknown_engine`` (non-retryable). The
``isinstance(exc, UnknownEngineError)`` dispatch makes the mapping
independent of message wording.
"""

from __future__ import annotations

import httpx
import pytest

from imagecli.engine import UnknownEngineError
from imagecli.nats.validators import _map_exception_to_error, _sanitize_delivery_exception


def test_unknown_engine_dispatch_isinstance_not_substring() -> None:
    """A UnknownEngineError whose message has been rewritten still routes to
    ``unknown_engine`` — the dispatcher uses ``isinstance``, not a substring
    match on str(exc)."""
    exc = UnknownEngineError("totally-rewritten message without the magic phrase")
    code, _detail = _map_exception_to_error(exc)
    assert code == "unknown_engine", (
        f"expected 'unknown_engine' from isinstance dispatch, got {code!r}; "
        "message-substring matching is the regression this test guards against."
    )


def test_unknown_engine_detail_built_from_available_attribute() -> None:
    """The dispatcher reads the structured ``available`` attribute, not the
    message string. Detail is the comma-joined registry list — no parsing of
    ``str(exc)`` involved."""
    exc = UnknownEngineError(
        "any message here",
        available=["flux2-klein", "sd35"],
    )
    code, detail = _map_exception_to_error(exc)
    assert code == "unknown_engine"
    assert detail == "flux2-klein, sd35"


def test_unknown_engine_detail_unknown_when_available_empty() -> None:
    """When the exception was raised without an ``available`` list (legacy
    callers or constructed in tests), the dispatcher reports ``unknown``
    rather than leaking the message."""
    exc = UnknownEngineError("some message")
    code, detail = _map_exception_to_error(exc)
    assert code == "unknown_engine"
    assert detail == "unknown"


def test_unknown_engine_detail_does_not_leak_message_content() -> None:
    """The dispatcher must NOT include attacker-controllable message
    fragments (e.g. paths, attacker-supplied engine names) in the wire
    detail — only the structured ``available`` list."""
    exc = UnknownEngineError(
        "Unknown engine '/etc/passwd'. Available: flux2-klein",
        available=["flux2-klein"],
    )
    _code, detail = _map_exception_to_error(exc)
    assert "/etc/passwd" not in detail
    assert detail == "flux2-klein"


def test_plain_value_error_does_not_route_to_unknown_engine() -> None:
    """A bare ``ValueError`` (not ``UnknownEngineError``) must NOT route to
    ``unknown_engine`` — only the typed subclass should."""
    exc = ValueError("Unknown engine 'foo'. Available: bar")
    code, _detail = _map_exception_to_error(exc)
    assert code == "generation_failed", (
        f"plain ValueError must fall through to generic mapping; got {code!r}. "
        "If this fails, dispatch is still substring-matching."
    )


# ── _sanitize_delivery_exception — secret-leak prevention (#97 B2/W4) ─────────


def test_sanitize_http_status_error_strips_url() -> None:
    """httpx.HTTPStatusError.str includes the request URL — sanitizer must NOT
    embed it in the returned string."""
    req = httpx.Request("PUT", "https://roxabituwer:8449/blobs?token=DEADBEEF")
    resp = httpx.Response(500, request=req)
    exc = httpx.HTTPStatusError("Server error '500' for url", request=req, response=resp)
    out = _sanitize_delivery_exception(exc)
    assert out == "upstream HTTP 500"
    assert "roxabituwer" not in out
    assert "DEADBEEF" not in out
    assert "?token=" not in out


def test_sanitize_connect_error_fixed_string() -> None:
    """ConnectError.str includes host/port — sanitizer collapses to a fixed
    diagnostic string."""
    exc = httpx.ConnectError("All connection attempts failed to roxabituwer:8449")
    out = _sanitize_delivery_exception(exc)
    assert out == "BlobStore connection failed"
    assert "roxabituwer" not in out


def test_sanitize_timeout_fixed_string() -> None:
    exc = httpx.ReadTimeout("timed out reading from /blobs?secret=ABCDEF")
    out = _sanitize_delivery_exception(exc)
    assert out == "BlobStore request timed out"
    assert "ABCDEF" not in out


def test_sanitize_generic_httperror_collapses() -> None:
    """A non-{Status,Timeout,Connect} httpx.HTTPError still gets collapsed —
    no message text reaches the wire."""
    exc = httpx.RemoteProtocolError("server disconnected before sending payload")
    out = _sanitize_delivery_exception(exc)
    assert out == "BlobStore transport error"


def test_sanitize_unknown_exception_returns_generic() -> None:
    """Non-httpx exceptions (e.g., asyncio errors) fall through to the
    generic sentinel."""
    exc = RuntimeError("kaboom — token=ABC123")
    out = _sanitize_delivery_exception(exc)
    assert out == "internal delivery error"
    assert "ABC123" not in out


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 502, 503])
def test_sanitize_http_status_codes_round_trip(status: int) -> None:
    req = httpx.Request("PUT", "http://h:9/blobs")
    resp = httpx.Response(status, request=req)
    exc = httpx.HTTPStatusError("x", request=req, response=resp)
    assert _sanitize_delivery_exception(exc) == f"upstream HTTP {status}"
