"""Canonical output paths for imagecli — layout strings + helpers.

Convention from `~/projects/CLAUDE.md`: code @ `~/projects/<tool>/`,
data @ `~/.roxabi/<tool>/`. The local CLI output dir sits under
`~/.roxabi/imagecli/out/` and is Syncthing-replicated between M₁ and M₂.

The NATS satellite (`nats/adapter.py`) no longer writes images to a shared
filesystem — issue #97 migrated the response surface to `BlobRef` via
`HttpBlobStore` (ADR-067). The `nats_output_dir` / `move_to_nats_output` /
`NATS_OUTPUT_DIR` / `IMAGECLI_NATS_OUTPUT_DIR` symbols and env var were
removed in that migration.
"""

from __future__ import annotations

__all__ = [
    "CLI_OUTPUT_DIR",
]

# Unexpanded literal — user-readable, suitable as a default in imagecli.toml
# and in user-facing messages. Callers expand at use time via `Path.expanduser()`.
CLI_OUTPUT_DIR = "~/.roxabi/imagecli/out"
