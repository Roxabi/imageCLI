"""Tests for imagecli.commands._helpers path-resolution utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagecli.commands._helpers import resolve_face_image, resolve_face_images, resolve_loras
from imagecli.lora_spec import LoraSpec


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


# ---------------------------------------------------------------------------
# resolve_loras — T17 (RED) + T20 (RED-GATE)
# ---------------------------------------------------------------------------

FM_LORAS = [
    LoraSpec(path="/models/a.safetensors", scale=0.8, trigger="tok_a"),
    LoraSpec(path="/models/b.safetensors", scale=1.2, trigger=None),
]


def test_resolve_loras_empty_cli_returns_fm() -> None:
    """Empty CLI flags → frontmatter list returned unchanged."""
    result = resolve_loras([], [], [], [], FM_LORAS)
    assert result == FM_LORAS


def test_resolve_loras_cli_replaces_fm() -> None:
    """CLI has 1 lora + FM has 2 → 1-element list (replace, not merge)."""
    result = resolve_loras(["/models/c.safetensors"], [], [], [], FM_LORAS)
    assert len(result) == 1
    assert result[0].path == "/models/c.safetensors"
    assert result[0].scale == 1.0  # default
    assert result[0].trigger is None


def test_resolve_loras_multi_cli_pairwise_scales() -> None:
    """Two loras with explicit scales → correct pairwise assignment."""
    result = resolve_loras(
        ["/a.safetensors", "/b.safetensors"],
        [1.0, 0.5],
        [],
        [],
        [],
    )
    assert len(result) == 2
    assert result[0] == LoraSpec(path="/a.safetensors", scale=1.0)
    assert result[1] == LoraSpec(path="/b.safetensors", scale=0.5)


def test_resolve_loras_multi_cli_with_triggers_and_embeddings() -> None:
    """Full pairwise: loras + scales + triggers + embeddings."""
    result = resolve_loras(
        ["/a.safetensors", "/b.safetensors"],
        [1.5, 0.7],
        ["tok_a", "tok_b"],
        ["/emb_a.safetensors", "/emb_b.safetensors"],
        [],
    )
    assert result[0] == LoraSpec(
        path="/a.safetensors", scale=1.5, trigger="tok_a", embedding_path="/emb_a.safetensors"
    )
    assert result[1] == LoraSpec(
        path="/b.safetensors", scale=0.7, trigger="tok_b", embedding_path="/emb_b.safetensors"
    )


def test_resolve_loras_scale_mismatch_raises() -> None:
    """Pairwise length mismatch for scales → ValueError with clear message."""
    with pytest.raises(ValueError, match="lora-scale"):
        resolve_loras(["/a.safetensors", "/b.safetensors"], [1.0], [], [], [])


def test_resolve_loras_trigger_mismatch_raises() -> None:
    """Pairwise length mismatch for triggers → ValueError."""
    with pytest.raises(ValueError, match="trigger"):
        resolve_loras(["/a.safetensors", "/b.safetensors"], [], ["tok_a"], [], [])


def test_resolve_loras_embedding_mismatch_raises() -> None:
    """Pairwise length mismatch for embeddings → ValueError."""
    with pytest.raises(ValueError, match="embedding"):
        resolve_loras(["/a.safetensors", "/b.safetensors"], [], [], ["/emb.safetensors"], [])


def test_resolve_loras_n1_cli_byte_identical_to_pre_v5() -> None:
    """N=1 CLI path produces LoraSpec identical to what the pre-#34 path produced."""
    result = resolve_loras(
        ["/models/lyra.safetensors"],
        [1.0],
        ["lyraface"],
        ["/models/lyra_emb.safetensors"],
        [],
    )
    expected = LoraSpec(
        path="/models/lyra.safetensors",
        scale=1.0,
        trigger="lyraface",
        embedding_path="/models/lyra_emb.safetensors",
    )
    assert len(result) == 1
    assert result[0] == expected


def test_resolve_loras_empty_cli_empty_fm_returns_empty() -> None:
    """Both empty → empty list, no error."""
    result = resolve_loras([], [], [], [], [])
    assert result == []


def test_resolve_loras_empty_string_trigger_coerces_to_none() -> None:
    """Pairwise empty-string trigger → None (not "")."""
    result = resolve_loras(["/a.safetensors"], [], [""], [], [])
    assert result[0].trigger is None
