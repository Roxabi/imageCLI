"""Tests for CLI commands — uses typer test runner + mocked engine."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from imagecli.cli import app

runner = CliRunner()


def make_mock_engine(
    name: str = "flux2-klein", default_steps: int = 28, default_guidance: float = 3.5
):
    engine = MagicMock()
    engine.name = name
    engine.description = f"Mock {name}"
    engine.default_steps = default_steps
    engine.default_guidance = default_guidance
    engine.generate.side_effect = lambda *a, output_path, **kw: (
        output_path.parent.mkdir(parents=True, exist_ok=True) or output_path
    )
    return engine


# ── engines command ────────────────────────────────────────────────────────────


def test_engines_command_lists_all():
    result = runner.invoke(app, ["engines"])
    assert result.exit_code == 0
    assert "flux2-klein" in result.output
    assert "flux1-dev" in result.output
    assert "sd35" in result.output


def test_engines_command_shows_defaults():
    result = runner.invoke(app, ["engines"])
    assert "28" in result.output  # flux2-klein default_steps
    assert "3.5" in result.output  # guidance


# ── generate command ───────────────────────────────────────────────────────────


def test_generate_from_prompt(tmp_path):
    mock_engine = make_mock_engine()
    with (
        patch(
            "imagecli.cli._load_config",
            return_value={
                "engine": "flux2-klein",
                "width": 1024,
                "height": 1024,
                "format": "png",
                "output_dir": str(tmp_path),
            },
        ),
        patch(
            "imagecli.engine.get_engine_class",
            return_value=type("E", (), {"default_steps": 28, "default_guidance": 3.5}),
        ),
        patch("imagecli.engine.get_engine", return_value=mock_engine),
    ):
        result = runner.invoke(app, ["generate", "a cat in space"])
    assert result.exit_code == 0
    mock_engine.generate.assert_called_once()


def test_generate_uses_engine_default_steps(tmp_path):
    """When no steps in config or CLI, engine default_steps is used."""
    mock_engine = make_mock_engine(default_steps=28)
    captured_steps = {}

    def capture_generate(*a, steps, **kw):
        captured_steps["steps"] = steps
        out = kw["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    mock_engine.generate.side_effect = capture_generate

    with (
        patch(
            "imagecli.cli._load_config",
            return_value={
                "engine": "flux2-klein",
                "width": 1024,
                "height": 1024,
                "format": "png",
                "output_dir": str(tmp_path),
            },
        ),
        patch(
            "imagecli.engine.get_engine_class",
            return_value=type("E", (), {"default_steps": 28, "default_guidance": 3.5}),
        ),
        patch("imagecli.engine.get_engine", return_value=mock_engine),
    ):
        result = runner.invoke(app, ["generate", "a cat in space"])
    assert result.exit_code == 0
    assert captured_steps["steps"] == 28


def test_generate_cli_steps_override_engine_default(tmp_path):
    """CLI --steps flag overrides engine default."""
    mock_engine = make_mock_engine(default_steps=28)
    captured = {}

    def capture_generate(*a, steps, **kw):
        captured["steps"] = steps
        out = kw["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    mock_engine.generate.side_effect = capture_generate

    with (
        patch(
            "imagecli.cli._load_config",
            return_value={
                "engine": "flux2-klein",
                "width": 1024,
                "height": 1024,
                "format": "png",
                "output_dir": str(tmp_path),
            },
        ),
        patch(
            "imagecli.engine.get_engine_class",
            return_value=type("E", (), {"default_steps": 28, "default_guidance": 3.5}),
        ),
        patch("imagecli.engine.get_engine", return_value=mock_engine),
    ):
        result = runner.invoke(app, ["generate", "a cat in space", "--steps", "60"])
    assert result.exit_code == 0
    assert captured["steps"] == 60


def test_generate_from_md_file(tmp_path):
    md = tmp_path / "prompt.md"
    md.write_text(
        textwrap.dedent("""\
        ---
        engine: flux1-dev
        width: 1280
        height: 720
        steps: 30
        ---

        A scenic mountain valley.
    """)
    )

    mock_engine = make_mock_engine("flux1-dev", default_steps=50, default_guidance=3.5)
    captured = {}

    def capture(*a, steps, width, height, **kw):
        captured.update(steps=steps, width=width, height=height)
        out = kw["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    mock_engine.generate.side_effect = capture

    with (
        patch(
            "imagecli.cli._load_config",
            return_value={
                "engine": "flux2-klein",
                "width": 1024,
                "height": 1024,
                "format": "png",
                "output_dir": str(tmp_path),
            },
        ),
        patch(
            "imagecli.engine.get_engine_class",
            return_value=type("E", (), {"default_steps": 50, "default_guidance": 3.5}),
        ),
        patch("imagecli.engine.get_engine", return_value=mock_engine),
    ):
        result = runner.invoke(app, ["generate", str(md)])
    assert result.exit_code == 0
    assert captured["steps"] == 30  # from frontmatter
    assert captured["width"] == 1280


# ── batch command ──────────────────────────────────────────────────────────────


def test_batch_no_md_files(tmp_path):
    with patch(
        "imagecli.cli._load_config",
        return_value={"engine": "flux2-klein", "width": 1024, "height": 1024, "format": "png", "output_dir": str(tmp_path)},
    ):
        result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 1
    assert "No .md files" in result.output


def test_batch_processes_all_md_files(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    out_dir = tmp_path / "out"

    for i in range(3):
        (prompt_dir / f"prompt{i}.md").write_text(f"Prompt {i}.\n")

    mock_engine = make_mock_engine()

    def generate(*a, output_path, **kw):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    mock_engine.generate.side_effect = generate

    with (
        patch(
            "imagecli.cli._load_config",
            return_value={
                "engine": "flux2-klein",
                "width": 1024,
                "height": 1024,
                "format": "png",
                "output_dir": str(out_dir),
            },
        ),
        patch(
            "imagecli.engine.get_engine_class",
            return_value=type("E", (), {"default_steps": 28, "default_guidance": 3.5}),
        ),
        patch("imagecli.engine.get_engine", return_value=mock_engine),
    ):
        result = runner.invoke(app, ["batch", str(prompt_dir)])
    assert result.exit_code == 0
    assert mock_engine.generate.call_count == 3
    assert "3 succeeded" in result.output


# ── info command ───────────────────────────────────────────────────────────────


def test_info_command(tmp_path):
    with patch("imagecli.cli._load_config", return_value={"engine": "flux2-klein", "width": 1024}):
        result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "flux2-klein" in result.output
