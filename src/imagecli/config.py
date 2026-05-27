"""TOML config loader — reads imagecli.toml by walking up from CWD to $HOME."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from imagecli.paths import CLI_OUTPUT_DIR

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

_FILENAME = "imagecli.toml"
_DEFAULTS: dict = {
    "engine": "flux2-klein",
    "width": 1024,
    "height": 1024,
    "steps": 50,
    "guidance": 4.0,
    "output_dir": CLI_OUTPUT_DIR,
    "format": "png",
    "quality": 95,
}

_DEFAULT_WEIGHTS_DIR = Path.home() / ".roxabi" / "imagecli" / "weights"


def _find_config() -> Path | None:
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    for path in [cwd, *cwd.parents]:
        candidate = path / _FILENAME
        if candidate.exists():
            return candidate
        if path == home:
            break
    return None


def load_config() -> dict:
    path = _find_config()
    if path is None:
        warnings.warn(
            f"{_FILENAME} not found — using built-in defaults. "
            f"Copy imagecli.example.toml to {Path.home() / _FILENAME} to configure.",
            stacklevel=2,
        )
        return dict(_DEFAULTS)

    with path.open("rb") as f:
        raw = tomllib.load(f)

    cfg = dict(_DEFAULTS)
    cfg.update(raw.get("defaults", {}))
    cfg["_config_path"] = str(path)
    return cfg


_BLOBSTORE_DEFAULT_ENDPOINT = "http://roxabituwer:8449"


def load_blobstore_config() -> dict[str, str | None]:
    """Resolve HttpBlobStore endpoint + token.

    Precedence: ``imagecli.toml [blobstore]`` > env (``IMAGECLI_BLOBSTORE_URL``,
    ``IMAGECLI_BLOBSTORE_TOKEN``) > default (endpoint = M₁ lyra-blobstore on
    port 8449; token = None — caller fails fast).

    Returns a dict with keys ``endpoint`` (always a str) and ``token`` (str or
    None). Callers wiring the worker MUST treat ``token is None`` as fatal.
    """
    raw: dict[str, object] = {}
    path = _find_config()
    if path is not None:
        with path.open("rb") as f:
            raw = tomllib.load(f).get("blobstore", {}) or {}

    endpoint_raw = raw.get("endpoint") or os.environ.get("IMAGECLI_BLOBSTORE_URL")
    endpoint = str(endpoint_raw) if endpoint_raw else _BLOBSTORE_DEFAULT_ENDPOINT

    token_raw = raw.get("token") or os.environ.get("IMAGECLI_BLOBSTORE_TOKEN")
    token = str(token_raw) if token_raw else None

    return {"endpoint": endpoint, "token": token}


def get_weights_dir() -> Path:
    """Return the weights directory from config [paths] section, expanding ~.

    Falls back to ~/.roxabi/imagecli/weights/ if not configured or no config file.
    Priority: CLI flag (N/A for paths) > TOML [paths].weights_dir > default.
    """
    config_path = _find_config()
    if config_path is None:
        return _DEFAULT_WEIGHTS_DIR

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    weights_str = raw.get("paths", {}).get("weights_dir")
    if weights_str:
        return Path(weights_str).expanduser()
    return _DEFAULT_WEIGHTS_DIR
