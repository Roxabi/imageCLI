"""Tests for imagecli.cli — CLI commands via CliRunner, _resolve_output."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from imagecli.cli import _resolve_output, app

runner = CliRunner()


# ── CLI help / smoke tests ──────────────────────────────────────────────────


def test_generate_help():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "prompt" in result.output.lower() or "PROMPT" in result.output


def test_engines_command():
    result = runner.invoke(app, ["engines"])
    assert result.exit_code == 0
    assert "flux2-klein" in result.output
    assert "flux1-dev" in result.output
    assert "sd35" in result.output


def test_info_command_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    # Should show config table regardless of GPU presence
    assert "engine" in result.output.lower() or "config" in result.output.lower()


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # no_args_is_help=True causes Typer to exit with code 0 or 2
    assert result.exit_code in (0, 2)
    assert "generate" in result.output
    assert "batch" in result.output


# ── _resolve_output ─────────────────────────────────────────────────────────


def test_resolve_output_basic(tmp_path: Path):
    cfg = {"output_dir": str(tmp_path)}
    result = _resolve_output(cfg, "image", "png", None)
    assert result == tmp_path / "image.png"


def test_resolve_output_no_clobber(tmp_path: Path):
    cfg = {"output_dir": str(tmp_path)}
    # Create the first file so the function must add a suffix
    (tmp_path / "image.png").touch()
    result = _resolve_output(cfg, "image", "png", None)
    assert result == tmp_path / "image_1.png"


def test_resolve_output_multiple_collisions(tmp_path: Path):
    cfg = {"output_dir": str(tmp_path)}
    (tmp_path / "image.png").touch()
    (tmp_path / "image_1.png").touch()
    (tmp_path / "image_2.png").touch()
    result = _resolve_output(cfg, "image", "png", None)
    assert result == tmp_path / "image_3.png"


def test_resolve_output_custom_dir(tmp_path: Path):
    cfg = {"output_dir": "ignored"}
    custom = tmp_path / "custom_out"
    result = _resolve_output(cfg, "test", "jpg", str(custom))
    assert result.parent == custom
    assert result.name == "test.jpg"


def test_resolve_output_creates_dir(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c"
    cfg = {"output_dir": str(nested)}
    result = _resolve_output(cfg, "img", "webp", None)
    assert nested.exists()
    assert result == nested / "img.webp"


def test_resolve_output_strips_dot_from_format(tmp_path: Path):
    cfg = {"output_dir": str(tmp_path)}
    result = _resolve_output(cfg, "photo", ".png", None)
    assert result.name == "photo.png"  # not photo..png


# ── generate command with mocked engine ─────────────────────────────────────


@patch("imagecli.cli._run_generate")
def test_generate_text_prompt(mock_run):
    mock_run.return_value = Path("/fake/output.png")
    result = runner.invoke(app, ["generate", "a cat in space"])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    args = mock_run.call_args
    assert args[0][0] == "a cat in space"  # prompt


@patch("imagecli.cli._run_generate")
def test_generate_with_engine_flag(mock_run):
    mock_run.return_value = Path("/fake/output.png")
    result = runner.invoke(app, ["generate", "test prompt", "-e", "flux1-dev"])
    assert result.exit_code == 0
    args = mock_run.call_args
    assert args[0][2] == "flux1-dev"  # engine_name


@patch("imagecli.cli._run_generate")
def test_generate_with_size_flags(mock_run):
    mock_run.return_value = Path("/fake/output.png")
    result = runner.invoke(app, ["generate", "test", "-W", "1280", "-H", "720"])
    assert result.exit_code == 0
    args = mock_run.call_args
    assert args[0][3] == 1280  # width
    assert args[0][4] == 720  # height


@patch("imagecli.cli._run_generate")
def test_generate_md_file(mock_run, tmp_path: Path):
    mock_run.return_value = Path("/fake/output.png")
    md_file = tmp_path / "prompt.md"
    md_file.write_text(
        "---\nengine: flux1-dev\nwidth: 512\nheight: 512\nsteps: 10\n"
        "guidance: 2.0\n---\n\nA test prompt.\n"
    )
    result = runner.invoke(app, ["generate", str(md_file)])
    assert result.exit_code == 0
    args = mock_run.call_args
    assert "test prompt" in args[0][0].lower()
    assert args[0][2] == "flux1-dev"
