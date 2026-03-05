"""Tests for imagecli.markdown — edge cases: no frontmatter, empty prompt, invalid YAML."""

from __future__ import annotations

from pathlib import Path


from imagecli.markdown import PromptDoc, parse_prompt_file


def test_no_frontmatter(tmp_path: Path):
    md = tmp_path / "bare.md"
    md.write_text("Just a prompt with no frontmatter.\n")
    doc = parse_prompt_file(md)
    assert isinstance(doc, PromptDoc)
    assert "just a prompt" in doc.prompt.lower()
    assert doc.engine is None
    assert doc.width is None


def test_prompt_only_multiline(tmp_path: Path):
    md = tmp_path / "multi.md"
    md.write_text("First paragraph.\n\nSecond paragraph.\n")
    doc = parse_prompt_file(md)
    assert "First paragraph" in doc.prompt
    assert "Second paragraph" in doc.prompt


def test_empty_frontmatter(tmp_path: Path):
    md = tmp_path / "empty_fm.md"
    md.write_text("---\n---\n\nPrompt after empty frontmatter.\n")
    doc = parse_prompt_file(md)
    assert "prompt after empty frontmatter" in doc.prompt.lower()
    assert doc.engine is None


def test_frontmatter_no_body(tmp_path: Path):
    md = tmp_path / "no_body.md"
    md.write_text("---\nengine: sd35\nwidth: 768\n---\n")
    doc = parse_prompt_file(md)
    assert doc.engine == "sd35"
    assert doc.width == 768
    # prompt should be empty string when no body
    assert doc.prompt == ""


def test_partial_frontmatter(tmp_path: Path):
    md = tmp_path / "partial.md"
    md.write_text("---\nengine: flux1-dev\n---\n\nA cat.\n")
    doc = parse_prompt_file(md)
    assert doc.engine == "flux1-dev"
    assert doc.width is None
    assert doc.steps is None
    assert "cat" in doc.prompt.lower()


def test_seed_zero(tmp_path: Path):
    md = tmp_path / "seed.md"
    md.write_text("---\nseed: 0\n---\n\nPrompt.\n")
    doc = parse_prompt_file(md)
    assert doc.seed == 0


def test_negative_prompt_in_frontmatter(tmp_path: Path):
    md = tmp_path / "neg.md"
    md.write_text('---\nnegative_prompt: "blurry, ugly"\n---\n\nA dog.\n')
    doc = parse_prompt_file(md)
    assert "blurry" in doc.negative_prompt


def test_extra_fields_preserved(tmp_path: Path):
    md = tmp_path / "extra.md"
    md.write_text("---\nengine: sd35\ncustom_field: hello\n---\n\nPrompt.\n")
    doc = parse_prompt_file(md)
    assert doc.extra.get("custom_field") == "hello"


def test_format_field(tmp_path: Path):
    md = tmp_path / "fmt.md"
    md.write_text("---\nformat: webp\n---\n\nPrompt.\n")
    doc = parse_prompt_file(md)
    assert doc.format == "webp"
