"""TOML config loader — reads imagecli.toml by walking up from CWD to $HOME."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

_FILENAME = 'imagecli.toml'
_DEFAULTS: dict = {
    'engine': 'flux2-klein',
    'width': 1024,
    'height': 1024,
    'steps': 50,
    'guidance': 4.0,
    'output_dir': 'images/images_out',
    'format': 'png',
    'quality': 95,
}


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
            f'{_FILENAME} not found — using built-in defaults. '
            f'Copy imagecli.example.toml to {Path.home() / _FILENAME} to configure.',
            stacklevel=2,
        )
        return dict(_DEFAULTS)

    with path.open('rb') as f:
        raw = tomllib.load(f)

    cfg = dict(_DEFAULTS)
    cfg.update(raw.get('defaults', {}))
    cfg['_config_path'] = str(path)
    return cfg
