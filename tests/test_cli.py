"""Tests for imagecli.cli — CLI commands via CliRunner, _resolve_output."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
    # Verify the no-CUDA branch was actually reached
    assert "no cuda" in result.output.lower() or "No CUDA GPU" in result.output


def test_info_command_shows_compute_capability():
    # Arrange: mock an Ampere RTX 3080 (sm_86) with 10 GB VRAM.
    mock_props = MagicMock()
    mock_props.name = "NVIDIA GeForce RTX 3080"
    mock_props.total_memory = 10 * 1024**3
    mock_props.major = 8
    mock_props.minor = 6
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=mock_props),
    ):
        # Act
        result = runner.invoke(app, ["info"])
    # Assert: GPU name, SM string, and architecture label all appear in output.
    assert result.exit_code == 0
    assert "RTX 3080" in result.output
    assert "sm_86" in result.output
    assert "Ampere" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # Typer's no_args_is_help=True exits 0 (Click) or 2 (Typer CliRunner) depending on version
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
    # Signature: _run_generate(prompt, neg, engine_name, w, h, s, g, sd, fmt, out_path, ...)
    # index:                       0      1       2       3  4  5  6   7    8       9
    args = mock_run.call_args
    assert args[0][2] == "flux1-dev"  # index 2 = engine_name


@patch("imagecli.cli._run_generate")
def test_generate_with_size_flags(mock_run):
    mock_run.return_value = Path("/fake/output.png")
    result = runner.invoke(app, ["generate", "test", "-W", "1280", "-H", "720"])
    assert result.exit_code == 0
    # Signature: _run_generate(prompt, neg, engine_name, w, h, s, g, sd, fmt, out_path, ...)
    # index:                       0      1       2       3  4  5  6   7    8       9
    args = mock_run.call_args
    assert args[0][3] == 1280  # index 3 = width
    assert args[0][4] == 720  # index 4 = height


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
    assert args[0][2] == "flux1-dev"  # index 2 = engine_name


@patch("imagecli.cli._run_generate")
def test_generate_explicit_output(mock_run, tmp_path: Path):
    mock_run.return_value = Path("/fake/output.png")
    custom_out = tmp_path / "custom.png"
    result = runner.invoke(app, ["generate", "test prompt", "-o", str(custom_out)])
    assert result.exit_code == 0
    # Signature: _run_generate(prompt, neg, engine_name, w, h, s, g, sd, fmt, out_path, ...)
    # index:                       0      1       2       3  4  5  6   7    8       9
    args = mock_run.call_args
    assert args[0][9] == custom_out  # index 9 = out_path


@patch("imagecli.cli._run_generate")
def test_generate_no_compile(mock_run):
    mock_run.return_value = Path("/fake/output.png")
    result = runner.invoke(app, ["generate", "test prompt", "--no-compile"])
    assert result.exit_code == 0
    assert mock_run.call_args.kwargs["compile"] is False


# ── _run_generate — Peak VRAM display ───────────────────────────────────────


def test_generate_shows_peak_vram(tmp_path: Path):
    # Arrange: a fake engine that returns a valid output path, with CUDA reporting
    # 8.5 GB peak reserved memory.
    fake_img = tmp_path / "out.png"
    fake_img.touch()

    mock_engine = MagicMock()
    mock_engine.name = "flux2-klein"
    mock_engine.description = "test engine"
    mock_engine.generate.return_value = fake_img

    with (
        patch("imagecli.engine.get_engine", return_value=mock_engine),
        patch("imagecli.engine.preflight_check"),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.max_memory_reserved", return_value=int(8.5 * 1024**3)),
    ):
        # Act
        result = runner.invoke(app, ["generate", "test prompt", "--output-dir", str(tmp_path)])

    # Assert: peak VRAM line is printed with the expected value.
    assert result.exit_code == 0
    assert "Peak VRAM reserved" in result.output
    assert "8.50 GB" in result.output
