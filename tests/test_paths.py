"""Tests for imagecli.paths — output dir resolution + traversal-safe move helper."""

from __future__ import annotations

from pathlib import Path

import pytest


# ── nats_output_dir ──────────────────────────────────────────────────────────


def test_nats_output_dir_default_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMAGECLI_NATS_OUTPUT_DIR", raising=False)
    from imagecli.paths import nats_output_dir

    out = nats_output_dir()
    assert out == Path.home() / ".roxabi" / "imagecli" / "nats_out"
    assert out.is_absolute()


def test_nats_output_dir_env_override_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGECLI_NATS_OUTPUT_DIR", str(tmp_path / "custom"))
    from imagecli.paths import nats_output_dir

    assert nats_output_dir() == tmp_path / "custom"


def test_nats_output_dir_env_override_tilde_expands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IMAGECLI_NATS_OUTPUT_DIR", "~/custom")
    from imagecli.paths import nats_output_dir

    assert nats_output_dir() == tmp_path / "custom"


# ── move_to_nats_output ──────────────────────────────────────────────────────


def _seed_tmp_file(tmp_path: Path, name: str = "src.png") -> Path:
    src = tmp_path / name
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    return src


def test_move_to_nats_output_returns_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGECLI_NATS_OUTPUT_DIR", str(tmp_path / "out"))
    from imagecli.paths import move_to_nats_output

    src = _seed_tmp_file(tmp_path)
    result = move_to_nats_output(src, "abcdef1234", "png")

    assert result.is_absolute()
    assert result.parent == (tmp_path / "out").resolve()
    assert result.name == "nats_abcdef12.png"
    assert not src.exists()  # source moved
    assert result.exists()


def test_move_to_nats_output_creates_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out"
    monkeypatch.setenv("IMAGECLI_NATS_OUTPUT_DIR", str(target))
    from imagecli.paths import move_to_nats_output

    src = _seed_tmp_file(tmp_path)
    result = move_to_nats_output(src, "12345678", "png")

    assert target.exists()
    assert result.parent == target.resolve()


def test_move_to_nats_output_blocks_request_id_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Defence-in-depth: even if validators are bypassed, the move helper must reject."""
    monkeypatch.setenv("IMAGECLI_NATS_OUTPUT_DIR", str(tmp_path / "out"))
    from imagecli.paths import move_to_nats_output

    src = _seed_tmp_file(tmp_path)
    with pytest.raises(ValueError, match="path escape"):
        move_to_nats_output(src, "a/../../b", "png")
    assert src.exists()  # not moved


def test_move_to_nats_output_blocks_fmt_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IMAGECLI_NATS_OUTPUT_DIR", str(tmp_path / "out"))
    from imagecli.paths import move_to_nats_output

    src = _seed_tmp_file(tmp_path)
    with pytest.raises(ValueError, match="path escape"):
        move_to_nats_output(src, "abcdef12", "png/../../etc")
    assert src.exists()


# ── CLI / NATS layout invariants ─────────────────────────────────────────────


def test_layout_constants_are_under_roxabi() -> None:
    from imagecli.paths import CLI_OUTPUT_DIR, NATS_OUTPUT_DIR

    assert CLI_OUTPUT_DIR.startswith("~/.roxabi/imagecli/")
    assert NATS_OUTPUT_DIR.startswith("~/.roxabi/imagecli/")
    assert CLI_OUTPUT_DIR != NATS_OUTPUT_DIR
