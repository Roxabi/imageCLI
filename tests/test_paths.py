"""Tests for imagecli.paths — local CLI output dir layout.

The NATS satellite no longer writes to a shared filesystem (issue #97 /
ADR-067). `nats_output_dir`, `move_to_nats_output`, and `NATS_OUTPUT_DIR`
were removed; the tests that exercised them were dropped with this file.
"""

from __future__ import annotations


def test_layout_constant_is_under_roxabi() -> None:
    from imagecli.paths import CLI_OUTPUT_DIR

    assert CLI_OUTPUT_DIR.startswith("~/.roxabi/imagecli/")
