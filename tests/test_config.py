"""Tests for imagecli.config — TOML config loader."""

from __future__ import annotations

import warnings
from unittest.mock import patch

from imagecli.config import load_config


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
    assert cfg["output_dir"] == "images/images_out"
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
