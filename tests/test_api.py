"""Tests for the imagecli public API surface."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_all_exports_importable():
    """Every name in imagecli.__all__ is accessible via getattr(imagecli, name)."""
    # Arrange
    import imagecli

    # Act + Assert
    for name in imagecli.__all__:
        assert hasattr(imagecli, name), (
            f"imagecli.__all__ lists {name!r} but it is not accessible via getattr"
        )
        obj = getattr(imagecli, name)
        assert obj is not None, f"{name!r} exported but is None"


def test_all_matches_public_names():
    """set(imagecli.__all__) equals the set of public (non-underscore) module-level names.

    'Module-level names' excludes imported submodules and stdlib names — it covers
    only the symbols explicitly re-exported as the public API (functions, classes, etc.).
    """
    # Arrange
    import types
    import imagecli

    # Act: collect names from module vars that are not private and not bare modules,
    # and whose defining module is imagecli or one of its subpackages.
    # This excludes stdlib imports (Path, warnings) and __future__ names (annotations)
    # that appear in vars() but are not part of the public API.
    public_names = {
        name
        for name, obj in vars(imagecli).items()
        if not name.startswith("_")
        and not isinstance(obj, types.ModuleType)
        and getattr(obj, "__module__", "").startswith("imagecli")
    }
    all_names = set(imagecli.__all__)

    # Assert: every name in __all__ is public
    assert all_names <= public_names, f"Names in __all__ but not public: {all_names - public_names}"
    # Assert: every public name is in __all__
    assert public_names <= all_names, (
        f"Public names not listed in __all__: {public_names - all_names}"
    )


def test_no_cli_imports_at_module_level():
    """Importing imagecli does NOT cause typer or rich to appear in sys.modules."""
    # Arrange: snapshot imagecli.* modules so we can restore after the test.
    # Without restore, the wipe creates a module identity split that breaks
    # patches in subsequent tests (they patch V2 objects while test fixtures
    # hold references to V1 objects).
    snapshot = {
        k: v for k, v in sys.modules.items() if k == "imagecli" or k.startswith("imagecli.")
    }
    try:
        keys_to_remove = list(snapshot.keys())
        for k in keys_to_remove:
            del sys.modules[k]

        # Record which typer/rich modules were already in sys.modules before the
        # re-import so we can detect newly introduced ones.
        cli_keys_before = {
            k
            for k in sys.modules
            if k == "typer" or k.startswith("typer.") or k == "rich" or k.startswith("rich.")
        }

        # Act
        import imagecli  # noqa: F401  (re-import after cache clear)

        # Assert
        new_cli_keys = {
            k
            for k in sys.modules
            if (k == "typer" or k.startswith("typer.") or k == "rich" or k.startswith("rich."))
            and k not in cli_keys_before
        }
        assert not new_cli_keys, (
            f"Importing imagecli introduced CLI-only modules into sys.modules: {new_cli_keys}"
        )
    finally:
        # Restore: remove any fresh imagecli.* entries from the re-import,
        # then put back the original module objects.
        for k in list(sys.modules):
            if k == "imagecli" or k.startswith("imagecli."):
                del sys.modules[k]
        sys.modules.update(snapshot)


def test_generate_signature():
    """generate() has the expected parameters with correct defaults."""
    # Arrange
    import imagecli

    sig = inspect.signature(imagecli.generate)
    params = sig.parameters

    # Act + Assert: required positional parameter
    assert "prompt" in params, "generate() is missing the 'prompt' parameter"
    assert params["prompt"].default is inspect.Parameter.empty, (
        "'prompt' should have no default (required)"
    )

    # Parameters that must default to None
    none_default_params = [
        "engine",
        "width",
        "height",
        "steps",
        "guidance",
        "seed",
        "output_path",
        "output_dir",
        "format",
    ]
    for name in none_default_params:
        assert name in params, f"generate() is missing the '{name}' parameter"
        assert params[name].default is None, (
            f"'{name}' should default to None, got {params[name].default!r}"
        )

    # negative_prompt defaults to "" (empty string)
    assert "negative_prompt" in params, "generate() is missing the 'negative_prompt' parameter"
    assert params["negative_prompt"].default == "", (
        f"'negative_prompt' should default to '', got {params['negative_prompt'].default!r}"
    )

    # compile defaults to True
    assert "compile" in params, "generate() is missing the 'compile' parameter"
    assert params["compile"].default is True, (
        f"'compile' should default to True, got {params['compile'].default!r}"
    )


def test_generate_is_callable():
    """imagecli.generate is callable."""
    # Arrange
    import imagecli

    # Act + Assert
    assert callable(imagecli.generate), "imagecli.generate is not callable"


def test_generate_calls_engine_and_cleanup():
    """generate() orchestrates preflight, engine.generate, and cleanup in finally."""
    # Arrange
    import imagecli

    mock_engine = MagicMock()
    mock_engine.generate.return_value = Path("/fake/out.png")

    with (
        patch.object(imagecli, "get_engine", return_value=mock_engine) as mock_get,
        patch.object(imagecli, "preflight_check") as mock_preflight,
        patch("imagecli._utils.resolve_output", return_value=Path("/fake/out.png")),
    ):
        # Act
        result = imagecli.generate("a test prompt")

    # Assert: returns the path from engine.generate
    assert result == Path("/fake/out.png")
    # Assert: engine was instantiated with default config engine name
    mock_get.assert_called_once()
    # Assert: preflight was called before generation
    mock_preflight.assert_called_once_with(mock_engine)
    # Assert: engine.generate was called with the prompt
    mock_engine.generate.assert_called_once()
    assert mock_engine.generate.call_args.args[0] == "a test prompt"
    # Assert: cleanup was called (even though no error occurred)
    mock_engine.cleanup.assert_called_once()


def test_generate_cleanup_called_on_failure():
    """generate() calls cleanup even when engine.generate raises."""
    # Arrange
    import imagecli

    mock_engine = MagicMock()
    mock_engine.generate.side_effect = RuntimeError("generation failed")

    with (
        patch.object(imagecli, "get_engine", return_value=mock_engine),
        patch.object(imagecli, "preflight_check"),
        patch("imagecli._utils.resolve_output", return_value=Path("/fake/out.png")),
    ):
        # Act
        try:
            imagecli.generate("a test prompt")
        except RuntimeError:
            pass

    # Assert: cleanup was still called despite the failure
    mock_engine.cleanup.assert_called_once()
