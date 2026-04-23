"""Tests for imagecli.commands._helpers path-resolution utilities."""

from __future__ import annotations

from pathlib import Path

from imagecli.commands._helpers import resolve_face_image, resolve_face_images


def test_resolve_face_image_absolute_path_passthrough(tmp_path: Path) -> None:
    abs_path = tmp_path / "face.png"
    abs_path.write_bytes(b"")
    result = resolve_face_image(str(abs_path), tmp_path / "prompts")
    assert result == str(abs_path)


def test_resolve_face_image_relative_resolved_against_prompt_dir(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "face.png").write_bytes(b"")
    result = resolve_face_image("face.png", prompt_dir)
    assert result == str((prompt_dir / "face.png").resolve())


def test_resolve_face_image_none_passthrough() -> None:
    assert resolve_face_image(None, Path("/tmp")) is None


def test_resolve_face_image_empty_string_passthrough() -> None:
    assert resolve_face_image("", Path("/tmp")) is None


def test_resolve_face_images_list_mixed(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    abs_path = tmp_path / "abs.png"
    abs_path.write_bytes(b"")
    (prompt_dir / "rel.png").write_bytes(b"")

    result = resolve_face_images([str(abs_path), "rel.png"], prompt_dir)
    assert result == [str(abs_path), str((prompt_dir / "rel.png").resolve())]


def test_resolve_face_images_none_passthrough() -> None:
    assert resolve_face_images(None, Path("/tmp")) is None


def test_resolve_face_images_empty_list_returns_none() -> None:
    assert resolve_face_images([], Path("/tmp")) is None
