"""Tests for imagecli.config — TOML config loader."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from imagecli.config import load_blobstore_config, load_config


_EXPECTED_KEYS = {
    "engine",
    "width",
    "height",
    "steps",
    "guidance",
    "output_dir",
    "format",
    "quality",
}


def test_defaults():
    # Arrange: no config file on disk
    with patch("imagecli.config._find_config", return_value=None):
        # Act
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config()

    # Assert: a warning was issued
    assert len(caught) == 1
    assert "imagecli.toml" in str(caught[0].message)

    # Assert: default values are correct
    assert cfg["engine"] == "flux2-klein"
    assert cfg["width"] == 1024
    assert cfg["height"] == 1024
    assert cfg["steps"] == 50
    assert cfg["guidance"] == 4.0
    assert cfg["output_dir"] == "~/.roxabi/imagecli/out"
    assert cfg["format"] == "png"
    assert cfg["quality"] == 95


def test_config_keys():
    # Arrange: no config file on disk
    with patch("imagecli.config._find_config", return_value=None):
        # Act
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = load_config()

    # Assert: all expected keys are present
    for key in _EXPECTED_KEYS:
        assert key in cfg, f"Missing expected config key: {key!r}"


# ── load_blobstore_config (#97) ──────────────────────────────────────────────


def test_load_blobstore_default_endpoint_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """No toml + no env → default endpoint, token=None (caller fails fast)."""
    monkeypatch.delenv("IMAGECLI_BLOBSTORE_URL", raising=False)
    monkeypatch.delenv("IMAGECLI_BLOBSTORE_TOKEN", raising=False)
    with patch("imagecli.config._find_config", return_value=None):
        cfg = load_blobstore_config()
    assert cfg["endpoint"] == "http://roxabituwer:8449"
    assert cfg["token"] is None


def test_load_blobstore_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env vars are honored when toml is absent."""
    monkeypatch.setenv("IMAGECLI_BLOBSTORE_URL", "http://envhost:9000")
    monkeypatch.setenv("IMAGECLI_BLOBSTORE_TOKEN", "env-token-abc")
    with patch("imagecli.config._find_config", return_value=None):
        cfg = load_blobstore_config()
    assert cfg["endpoint"] == "http://envhost:9000"
    assert cfg["token"] == "env-token-abc"


def test_load_blobstore_from_toml_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """toml [blobstore] wins over env vars."""
    monkeypatch.setenv("IMAGECLI_BLOBSTORE_URL", "http://envhost:9000")
    monkeypatch.setenv("IMAGECLI_BLOBSTORE_TOKEN", "env-token-abc")
    toml_path = tmp_path / "imagecli.toml"
    toml_path.write_text(
        '[blobstore]\nendpoint = "http://tomlhost:1234"\ntoken = "toml-token-xyz"\n'
    )
    with patch("imagecli.config._find_config", return_value=toml_path):
        cfg = load_blobstore_config()
    assert cfg["endpoint"] == "http://tomlhost:1234"
    assert cfg["token"] == "toml-token-xyz"


def test_load_blobstore_toml_partial_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """toml with endpoint but no token falls back to env for the missing field."""
    monkeypatch.delenv("IMAGECLI_BLOBSTORE_URL", raising=False)
    monkeypatch.setenv("IMAGECLI_BLOBSTORE_TOKEN", "env-only-token")
    toml_path = tmp_path / "imagecli.toml"
    toml_path.write_text('[blobstore]\nendpoint = "http://tomlhost:1234"\n')
    with patch("imagecli.config._find_config", return_value=toml_path):
        cfg = load_blobstore_config()
    assert cfg["endpoint"] == "http://tomlhost:1234"
    assert cfg["token"] == "env-only-token"


def test_load_blobstore_empty_blobstore_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """toml present without a [blobstore] section yields defaults."""
    monkeypatch.delenv("IMAGECLI_BLOBSTORE_URL", raising=False)
    monkeypatch.delenv("IMAGECLI_BLOBSTORE_TOKEN", raising=False)
    toml_path = tmp_path / "imagecli.toml"
    toml_path.write_text('[defaults]\nengine = "flux2-klein"\n')
    with patch("imagecli.config._find_config", return_value=toml_path):
        cfg = load_blobstore_config()
    assert cfg["endpoint"] == "http://roxabituwer:8449"
    assert cfg["token"] is None
