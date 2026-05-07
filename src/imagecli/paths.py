"""Canonical output paths for imagecli — layout strings + helpers.

Convention from `~/projects/CLAUDE.md`: code @ `~/projects/<tool>/`,
data @ `~/.roxabi/<tool>/`. Outputs sit under `~/.roxabi/imagecli/{out,nats_out}/`
and are Syncthing-replicated between M₁ and M₂.

Two callers, one source of truth: the CLI default (`config.py`) and the NATS
satellite (`nats/adapter.py`) both resolve their output dirs through here.
Adding a third caller (HTTP adapter, daemon recovery, ...) should reuse this
module rather than fork another path-resolution helper.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

__all__ = [
    "CLI_OUTPUT_DIR",
    "NATS_OUTPUT_DIR",
    "nats_output_dir",
    "move_to_nats_output",
]

# Unexpanded literals — user-readable, suitable as defaults in imagecli.toml
# and in user-facing messages. Callers expand at use time via `Path.expanduser()`.
CLI_OUTPUT_DIR = "~/.roxabi/imagecli/out"
NATS_OUTPUT_DIR = "~/.roxabi/imagecli/nats_out"


def nats_output_dir() -> Path:
    """NATS satellite output dir, expanded to an absolute path.

    Honors `IMAGECLI_NATS_OUTPUT_DIR` for daemons whose CWD makes a
    config-file walk-up unreliable.
    """
    override = os.environ.get("IMAGECLI_NATS_OUTPUT_DIR")
    raw = override or NATS_OUTPUT_DIR
    return Path(raw).expanduser()


def move_to_nats_output(tmp_path: Path, request_id: str, fmt: str) -> Path:
    """Move a generated tmp file to the NATS output dir; return the absolute resolved path.

    Caller MUST pre-validate `request_id` and `fmt` against the contract — see
    `imagecli.nats.validators._validate_request`. The `is_relative_to` check
    below is defence-in-depth: if validation is ever bypassed or weakened, an
    attempted path-escape via crafted `request_id` / `fmt` raises before
    anything is written.
    """
    out_dir = nats_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"nats_{request_id[:8]}.{fmt}"
    if not final.resolve().is_relative_to(out_dir.resolve()):
        raise ValueError(f"path escape detected: {final}")
    shutil.move(tmp_path, final)
    return final.resolve()
