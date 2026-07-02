"""Tests for imagecli.markdown — YAML frontmatter + body text parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagecli.lora_spec import LoraSpec
from imagecli.markdown import PromptDoc, parse_prompt_file


def test_parse_prompt_file(tmp_path: Path):
    # Arrange: create a temporary .md file with frontmatter and body
    md_file = tmp_path / "test_prompt.md"
    md_file.write_text(
        "---\n"
        "engine: flux2-klein\n"
        "width: 1280\n"
        "height: 720\n"
        "steps: 30\n"
        "guidance: 3.5\n"
        "negative_prompt: blurry, low quality\n"
        "---\n"
        "\n"
        "A beautiful sunset over the ocean.\n",
        encoding="utf-8",
    )

    # Act
    doc = parse_prompt_file(md_file)

    # Assert: returns a PromptDoc instance
    assert isinstance(doc, PromptDoc)

    # Assert: frontmatter fields are parsed correctly
    assert doc.engine == "flux2-klein"
    assert doc.width == 1280
    assert doc.height == 720
    assert doc.steps == 30
    assert doc.guidance == pytest.approx(3.5)
    assert "blurry" in doc.negative_prompt
    assert "low quality" in doc.negative_prompt

    # Assert: body text is the prompt
    assert "sunset" in doc.prompt
    assert "ocean" in doc.prompt

    # No LoRA → empty list, singular aliases are defaults
    assert doc.loras == []
    assert doc.lora_path is None
    assert doc.lora_scale == pytest.approx(1.0)
    assert doc.trigger is None
    assert doc.embedding_path is None


# ---------------------------------------------------------------------------
# V4 — loras: list parsing
# ---------------------------------------------------------------------------


def test_loras_list_two_entries(tmp_path: Path):
    md_file = tmp_path / "multi_lora.md"
    md_file.write_text(
        "---\n"
        "loras:\n"
        "  - path: /models/a.safetensors\n"
        "    scale: 1.0\n"
        "    trigger: lyraface\n"
        "  - path: /models/b.safetensors\n"
        "    scale: 0.8\n"
        "    trigger: mickface\n"
        "    embedding_path: /models/mick_emb.safetensors\n"
        "---\n"
        "\n"
        "Portrait.\n",
        encoding="utf-8",
    )

    doc = parse_prompt_file(md_file)

    assert doc.loras == [
        LoraSpec(path="/models/a.safetensors", scale=1.0, trigger="lyraface"),
        LoraSpec(
            path="/models/b.safetensors",
            scale=0.8,
            trigger="mickface",
            embedding_path="/models/mick_emb.safetensors",
        ),
    ]
    # N != 1 → singular aliases reset to defaults
    assert doc.lora_path is None
    assert doc.lora_scale == pytest.approx(1.0)
    assert doc.trigger is None
    assert doc.embedding_path is None


def test_loras_list_single_entry_populates_singular_aliases(tmp_path: Path):
    md_file = tmp_path / "single_lora_list.md"
    md_file.write_text(
        "---\n"
        "loras:\n"
        "  - path: /models/solo.safetensors\n"
        "    scale: 1.2\n"
        "    trigger: sololora\n"
        "---\n"
        "\n"
        "Test prompt.\n",
        encoding="utf-8",
    )

    doc = parse_prompt_file(md_file)

    assert doc.loras == [LoraSpec(path="/models/solo.safetensors", scale=1.2, trigger="sololora")]
    assert doc.lora_path == "/models/solo.safetensors"
    assert doc.lora_scale == pytest.approx(1.2)
    assert doc.trigger == "sololora"
    assert doc.embedding_path is None


def test_singular_lora_kwargs_still_work(tmp_path: Path):
    md_file = tmp_path / "singular.md"
    md_file.write_text(
        "---\n"
        "lora_path: /models/x.safetensors\n"
        "lora_scale: 0.9\n"
        "trigger: xface\n"
        "embedding_path: /models/x_emb.safetensors\n"
        "---\n"
        "\n"
        "Singular LoRA.\n",
        encoding="utf-8",
    )

    doc = parse_prompt_file(md_file)

    assert doc.loras == [
        LoraSpec(
            path="/models/x.safetensors",
            scale=0.9,
            trigger="xface",
            embedding_path="/models/x_emb.safetensors",
        )
    ]
    assert doc.lora_path == "/models/x.safetensors"
    assert doc.lora_scale == pytest.approx(0.9)
    assert doc.trigger == "xface"
    assert doc.embedding_path == "/models/x_emb.safetensors"


def test_mixed_form_raises(tmp_path: Path):
    md_file = tmp_path / "mixed.md"
    md_file.write_text(
        "---\n"
        "loras:\n"
        "  - path: /models/a.safetensors\n"
        "lora_path: /models/b.safetensors\n"
        "---\n"
        "\n"
        "Mixed form.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot mix"):
        parse_prompt_file(md_file)


def test_empty_loras_key(tmp_path: Path):
    """An explicit empty loras: list → doc.loras == []."""
    md_file = tmp_path / "empty_loras.md"
    md_file.write_text(
        "---\nloras: []\n---\n\nNo LoRA.\n",
        encoding="utf-8",
    )

    doc = parse_prompt_file(md_file)
    assert doc.loras == []
    assert doc.lora_path is None


def test_missing_loras_key(tmp_path: Path):
    """No lora keys at all → doc.loras == []."""
    md_file = tmp_path / "no_lora.md"
    md_file.write_text(
        "---\nengine: flux2-klein\n---\n\nNo LoRA.\n",
        encoding="utf-8",
    )

    doc = parse_prompt_file(md_file)
    assert doc.loras == []


def test_loras_item_missing_path_raises(tmp_path: Path):
    md_file = tmp_path / "missing_path.md"
    md_file.write_text(
        "---\nloras:\n  - scale: 1.0\n    trigger: nopath\n---\n\nMissing path.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required key 'path'"):
        parse_prompt_file(md_file)
