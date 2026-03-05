"""Tests for imagecli.markdown — YAML frontmatter + body text parser."""

from __future__ import annotations

from pathlib import Path

import pytest

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
