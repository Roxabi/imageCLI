"""Tests for markdown.py — YAML frontmatter parser."""

from __future__ import annotations

import textwrap
from pathlib import Path


from imagecli.markdown import parse_prompt_file


def write_md(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.md"
    p.write_text(textwrap.dedent(content))
    return p


def test_full_frontmatter(tmp_path):
    f = write_md(
        tmp_path,
        """\
        ---
        engine: flux1-dev
        width: 1280
        height: 720
        steps: 30
        guidance: 3.5
        seed: 42
        negative_prompt: "blurry"
        format: webp
        ---

        A beautiful landscape.
    """,
    )
    doc = parse_prompt_file(f)
    assert doc.engine == "flux1-dev"
    assert doc.width == 1280
    assert doc.height == 720
    assert doc.steps == 30
    assert doc.guidance == 3.5
    assert doc.seed == 42
    assert doc.negative_prompt == "blurry"
    assert doc.format == "webp"
    assert doc.prompt == "A beautiful landscape."


def test_no_frontmatter(tmp_path):
    f = write_md(tmp_path, "Just a plain prompt.\n")
    doc = parse_prompt_file(f)
    assert doc.prompt == "Just a plain prompt."
    assert doc.engine is None
    assert doc.width is None
    assert doc.steps is None


def test_partial_frontmatter(tmp_path):
    f = write_md(
        tmp_path,
        """\
        ---
        engine: sd35
        seed: 7
        ---

        Minimal prompt.
    """,
    )
    doc = parse_prompt_file(f)
    assert doc.engine == "sd35"
    assert doc.seed == 7
    assert doc.width is None
    assert doc.guidance is None
    assert doc.prompt == "Minimal prompt."


def test_multiline_prompt(tmp_path):
    f = write_md(
        tmp_path,
        """\
        ---
        engine: flux2-klein
        ---

        First paragraph.

        Second paragraph.
    """,
    )
    doc = parse_prompt_file(f)
    assert "First paragraph." in doc.prompt
    assert "Second paragraph." in doc.prompt


def test_extra_frontmatter_keys_preserved(tmp_path):
    f = write_md(
        tmp_path,
        """\
        ---
        engine: flux2-klein
        custom_key: my_value
        ---

        Prompt.
    """,
    )
    doc = parse_prompt_file(f)
    assert doc.extra.get("custom_key") == "my_value"
