"""Tests for nats.validators._map_exception_to_error dispatch.

Guards the regression risk identified in audit Finding 5: if the registry
raise-site message changes, a substring-based dispatch silently
mis-routes unknown-engine errors to ``generation_failed`` (retryable),
when it should remain ``unknown_engine`` (non-retryable). The
``isinstance(exc, UnknownEngineError)`` dispatch makes the mapping
independent of message wording.
"""

from __future__ import annotations

from imagecli.engine import UnknownEngineError
from imagecli.nats.validators import _map_exception_to_error


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


def test_unknown_engine_detail_extracted_from_message_when_present() -> None:
    """When the message follows the original ``Unknown engine 'x'. Available: …``
    shape, the detail after the first ``: `` is returned (preserves the
    existing observable behaviour for legitimate raise sites)."""
    exc = UnknownEngineError("Unknown engine 'nope'. Available: flux2-klein, sd35")
    code, detail = _map_exception_to_error(exc)
    assert code == "unknown_engine"
    assert detail == "flux2-klein, sd35"


def test_plain_value_error_does_not_route_to_unknown_engine() -> None:
    """A bare ``ValueError`` (not ``UnknownEngineError``) must NOT route to
    ``unknown_engine`` — only the typed subclass should."""
    exc = ValueError("Unknown engine 'foo'. Available: bar")
    code, _detail = _map_exception_to_error(exc)
    assert code == "generation_failed", (
        f"plain ValueError must fall through to generic mapping; got {code!r}. "
        "If this fails, dispatch is still substring-matching."
    )
