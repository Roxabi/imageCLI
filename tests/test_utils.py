"""Tests for imagecli._utils — output path resolution + tilde expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagecli._utils import resolve_output


def test_resolve_output_expands_tilde_from_cfg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`~/...` in cfg["output_dir"] must be expanded against $HOME at use time."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = {"output_dir": "~/my_out"}

    path = resolve_output(cfg, "img", "png", output_dir=None)

    assert path == tmp_path / "my_out" / "img.png"
    assert (tmp_path / "my_out").is_dir()


def test_resolve_output_expands_tilde_from_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The --output-dir flag value must also expand `~/...`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = {"output_dir": "/should/be/ignored"}

    path = resolve_output(cfg, "img", "png", output_dir="~/flag_out")

    assert path == tmp_path / "flag_out" / "img.png"


def test_resolve_output_anti_clobber_suffix(tmp_path: Path) -> None:
    """When the candidate exists, suffix _1, _2, ... — never overwrite."""
    cfg = {"output_dir": str(tmp_path)}

    p1 = resolve_output(cfg, "x", "png", output_dir=None)
    p1.write_bytes(b"")
    p2 = resolve_output(cfg, "x", "png", output_dir=None)
    p2.write_bytes(b"")
    p3 = resolve_output(cfg, "x", "png", output_dir=None)

    assert p1.name == "x.png"
    assert p2.name == "x_1.png"
    assert p3.name == "x_2.png"
