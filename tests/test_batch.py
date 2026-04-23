"""Tests for imagecli.cli batch command — file discovery, engine caching, error handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from imagecli.cli import app
from imagecli.engine import ImageEngine

runner = CliRunner()


def _make_md(path: Path, engine: str = "flux2-klein", prompt: str = "test prompt") -> Path:
    path.write_text(
        f"---\nengine: {engine}\nwidth: 512\nheight: 512\nsteps: 4\n"
        f"guidance: 0.0\n---\n\n{prompt}\n"
    )
    return path


@patch("imagecli.commands._batch_sequential.run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_discovers_md_files(mock_get_engine, mock_run, tmp_path: Path):
    mock_get_engine.return_value = MagicMock(
        spec=ImageEngine, cleanup=MagicMock(), supports_two_phase=False
    )
    mock_run.return_value = Path("/fake/out.png")

    _make_md(tmp_path / "a.md")
    _make_md(tmp_path / "b.md")
    (tmp_path / "not_md.txt").write_text("ignored")

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert mock_run.call_count == 2


@patch("imagecli.commands._batch_sequential.run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_caches_engine(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock(spec=ImageEngine, supports_two_phase=False)
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
    # _run_generate must have been called once per file (3 files)
    assert mock_run.call_count == 3
    # All 3 calls must pass the same cached engine_instance
    for call in mock_run.call_args_list:
        assert call.kwargs["engine_instance"] is mock_engine
    # clear_cache called between each successful generation
    assert mock_engine.clear_cache.call_count == 3


@patch("imagecli.commands._batch_sequential.run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_no_compile(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock(spec=ImageEngine, supports_two_phase=False)
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.return_value = Path("/fake/out.png")

    _make_md(tmp_path / "a.md")

    result = runner.invoke(app, ["batch", str(tmp_path), "--no-compile"])
    assert result.exit_code == 0
    mock_get_engine.assert_called_once_with(
        "flux2-klein",
        compile=False,
        lora_path=None,
        lora_scale=1.0,
        trigger=None,
        embedding_path=None,
    )


def test_batch_empty_dir(tmp_path: Path):
    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 1
    assert "no .md files" in result.output.lower()


@patch("imagecli.commands._batch_sequential.run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_counts_failures(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock(spec=ImageEngine, supports_two_phase=False)
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine

    _make_md(tmp_path / "good.md", prompt="good prompt")
    _make_md(tmp_path / "bad.md", prompt="bad prompt")

    def side_effect(prompt, *args, **kwargs):
        if "bad prompt" in prompt:
            raise RuntimeError("generation failed")
        return Path("/fake/out.png")

    mock_run.side_effect = side_effect

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert "1 succeeded" in result.output
    assert "1 failed" in result.output


@patch("imagecli.commands._batch_sequential.run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_engine_override(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock(spec=ImageEngine, supports_two_phase=False)
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.return_value = Path("/fake/out.png")

    # File says flux2-klein, but CLI overrides to sd35
    _make_md(tmp_path / "a.md", engine="flux2-klein")

    result = runner.invoke(app, ["batch", str(tmp_path), "-e", "sd35"])
    assert result.exit_code == 0
    mock_get_engine.assert_called_once_with(
        "sd35",
        compile=True,
        lora_path=None,
        lora_scale=1.0,
        trigger=None,
        embedding_path=None,
    )


@patch("imagecli.commands._batch_sequential.run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_cleanup_called(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock(spec=ImageEngine, supports_two_phase=False)
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.return_value = Path("/fake/out.png")

    _make_md(tmp_path / "a.md")

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    mock_engine.cleanup.assert_called_once()


@patch("imagecli.commands._batch_sequential.run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_cleanup_after_failures(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock(spec=ImageEngine, supports_two_phase=False)
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.side_effect = RuntimeError("all fail")

    _make_md(tmp_path / "a.md")
    _make_md(tmp_path / "b.md")

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert "0 succeeded" in result.output
    assert "2 failed" in result.output
    # Cleanup is called on each failure to recover VRAM/RAM.
    # First failure triggers cleanup; engine is re-created for second file,
    # second failure triggers cleanup again. No final cleanup needed (already None).
    assert mock_engine.cleanup.call_count == 2


@patch("imagecli.commands._batch_sequential.run_generate")
@patch("imagecli.engine.get_engine")
def test_batch_shows_file_index(mock_get_engine, mock_run, tmp_path: Path):
    mock_engine = MagicMock(spec=ImageEngine, supports_two_phase=False)
    mock_engine.cleanup = MagicMock()
    mock_get_engine.return_value = mock_engine
    mock_run.return_value = Path("/fake/out.png")

    _make_md(tmp_path / "a.md")
    _make_md(tmp_path / "b.md")
    _make_md(tmp_path / "c.md")

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert "1/3" in result.output
    assert "2/3" in result.output
    assert "3/3" in result.output


@patch("imagecli.engine.get_engine")
def test_batch_two_phase(mock_get_engine, tmp_path: Path):
    """When engine supports 2-phase, batch encodes all then generates all."""
    mock_engine = MagicMock(spec=ImageEngine, supports_two_phase=True, _pipe=True)
    mock_engine.encode_prompt.return_value = {"prompt_embeds": "fake", "text_ids": "fake"}
    mock_engine.generate_from_embeddings.return_value = Path("/fake/out.png")
    mock_engine.effective_steps.return_value = 4
    mock_get_engine.return_value = mock_engine

    _make_md(tmp_path / "a.md")
    _make_md(tmp_path / "b.md")

    result = runner.invoke(app, ["batch", str(tmp_path)])
    assert result.exit_code == 0
    assert mock_engine.load_for_encode.call_count == 1
    assert mock_engine.encode_prompt.call_count == 2
    assert mock_engine.start_generation_phase.call_count == 1
    assert mock_engine.generate_from_embeddings.call_count == 2
    assert "2 succeeded" in result.output
