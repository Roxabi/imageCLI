"""Tests for imagecli.cli batch command — file discovery, engine caching, error handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from imagecli.cli import app

runner = CliRunner()


def _make_md(path: Path, engine: str = "flux2-klein", prompt: str = "test prompt") -> Path:
    path.write_text(
        f"---\nengine: {engine}\nwidth: 512\nheight: 512\nsteps: 4\n"
        f"guidance: 0.0\n---\n\n{prompt}\n"
    )
    return path


@patch("imagecli.cli._run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_discovers_md_files(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock()
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.return_value = Path("/fake/out.png")

    _make_md(tmp_path / "a.md")
    _make_md(tmp_path / "b.md")
    (tmp_path / "not_md.txt").write_text("ignored")

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert mock_run.call_count == 2


@patch("imagecli.cli._run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_caches_engine(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock()
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.return_value = Path("/fake/out.png")

    _make_md(tmp_path / "a.md", engine="flux2-klein")
    _make_md(tmp_path / "b.md", engine="flux2-klein")
    _make_md(tmp_path / "c.md", engine="flux2-klein")

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    # Engine should only be created once for the same engine name
    mock_get_engine.assert_called_once()


def test_batch_empty_dir(tmp_path: Path):
    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 1
    assert "no .md files" in result.output.lower()


@patch("imagecli.cli._run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_counts_failures(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock()
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine

    _make_md(tmp_path / "good.md")
    _make_md(tmp_path / "bad.md")

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # bad.md (sorted first alphabetically)
            raise RuntimeError("generation failed")
        return Path("/fake/out.png")

    mock_run.side_effect = side_effect

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 succeeded" in result.output
    assert "1 failed" in result.output


@patch("imagecli.cli._run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_engine_override(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock()
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.return_value = Path("/fake/out.png")

    # File says flux2-klein, but CLI overrides to sd35
    _make_md(tmp_path / "a.md", engine="flux2-klein")

    result = runner.invoke(app, ["batch", str(tmp_path), "-e", "sd35"])
    assert result.exit_code == 0
    mock_get_engine.assert_called_once_with("sd35", compile=True)


@patch("imagecli.cli._run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_cleanup_called(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock()
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.return_value = Path("/fake/out.png")

    _make_md(tmp_path / "a.md")

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    mock_engine.cleanup.assert_called_once()
