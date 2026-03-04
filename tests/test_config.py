"""Tests for config.py — TOML loader."""

from __future__ import annotations

import textwrap

import pytest

from imagecli.config import load_config


def test_defaults_returned_when_no_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load_config()
    assert cfg["engine"] == "flux2-klein"
    assert cfg["width"] == 1024
    assert cfg["height"] == 1024
    assert "steps" not in cfg
    assert "guidance" not in cfg
    assert cfg["format"] == "png"


def test_toml_overrides_defaults(tmp_path, monkeypatch):
    toml = tmp_path / "imagecli.toml"
    toml.write_text(
        textwrap.dedent("""\
        [defaults]
        engine = "flux1-dev"
        width = 1280
        height = 720
        steps = 30
        guidance = 3.0
    """)
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg["engine"] == "flux1-dev"
    assert cfg["width"] == 1280
    assert cfg["height"] == 720
    assert cfg["steps"] == 30
    assert cfg["guidance"] == 3.0


def test_config_path_recorded(tmp_path, monkeypatch):
    toml = tmp_path / "imagecli.toml"
    toml.write_text("[defaults]\n")
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert "_config_path" in cfg
    assert cfg["_config_path"] == str(toml)


def test_toml_walk_up(tmp_path, monkeypatch):
    """Config file in parent dir is found when CWD is a subdirectory."""
    toml = tmp_path / "imagecli.toml"
    toml.write_text("[defaults]\nwidth = 512\n")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load_config()
    assert cfg["width"] == 512


def test_warns_when_no_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.warns(UserWarning, match="imagecli.toml not found"):
        load_config()
