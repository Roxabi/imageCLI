"""Regression tests for socket-bound error sanitization in daemon_handlers.

Each of the 3 error paths (_handle_blend, _handle_encode, _handle_job) must
serialize exceptions via `roxabi_nats.sanitize_for_wire`, which:
  - scrubs URL userinfo for known credential-bearing schemes, and
  - truncates the message to DEFAULT_MAX_LEN (200) chars with a `…` marker.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from imagecli import daemon_handlers as h


@pytest.fixture
def captured(monkeypatch):
    """Capture (conn, payload) pairs that handlers pass to _send_json."""
    payloads: list[dict] = []
    monkeypatch.setattr(h, "_send_json", lambda conn, payload: payloads.append(payload))
    return payloads


def _error_payload(payloads: list[dict]) -> str:
    err = next((p["error"] for p in payloads if p.get("ok") is False), None)
    assert err is not None, f"no error payload found in {payloads!r}"
    return err


def test_blend_scrubs_credentials_in_error(captured, monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            OSError("nats://user:secret@host:4222/sub: denied")
        ),
    )

    # Act
    h._handle_blend(
        MagicMock(),
        {"inputs": [{"path": "x.pt", "weight": 1.0}], "out_path": str(tmp_path / "o.pt")},
    )

    # Assert
    err = _error_payload(captured)
    assert "user:secret" not in err
    assert "***:***" in err


def test_blend_truncates_long_error(captured, monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setattr(
        torch, "load", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("x" * 1000))
    )

    # Act
    h._handle_blend(
        MagicMock(),
        {"inputs": [{"path": "x.pt", "weight": 1.0}], "out_path": str(tmp_path / "o.pt")},
    )

    # Assert
    err = _error_payload(captured)
    assert len(err) <= 200
    assert err.endswith("…")


def test_encode_scrubs_credentials_in_error(captured, tmp_path):
    # Arrange
    encoder = MagicMock()
    encoder.encode_prompt.side_effect = OSError("nats://u:p@host:4222/sub: refused")
    req = {
        "jobs": [
            {"id": "j1", "prompt": "hi", "embed_path": str(tmp_path / "j1.pt")},
        ]
    }

    # Act
    h._handle_encode(MagicMock(), req, encoder)

    # Assert
    err = _error_payload(captured)
    assert "u:p" not in err
    assert "***:***" in err


def test_job_scrubs_credentials_in_error(captured, monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            OSError("nats+tls://admin:hunter2@cache:4222/0: gone")
        ),
    )
    req = {
        "action": "generate",
        "jobs": [
            {
                "id": "g1",
                "embed_path": str(tmp_path / "g1.pt"),
                "out_path": str(tmp_path / "g1.png"),
            }
        ],
    }

    # Act
    h._handle_job(MagicMock(), req, MagicMock())

    # Assert
    err = _error_payload(captured)
    assert "admin:hunter2" not in err
    assert "***:***" in err
