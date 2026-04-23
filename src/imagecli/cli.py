"""imagecli — unified CLI for local image generation.

Thin Typer wiring. Command implementations live in ``imagecli.commands``.
"""

from __future__ import annotations

import typer

from imagecli.commands import batch as batch_cmd
from imagecli.commands import generate as generate_cmd
from imagecli.commands import meta as meta_cmd
from imagecli.commands import serve as serve_cmd

app = typer.Typer(
    name="imagecli",
    help="Local image generation — FLUX.2-klein, FLUX.1-dev, SD3.5 backends.",
    no_args_is_help=True,
)

generate_cmd.register(app)
batch_cmd.register(app)
serve_cmd.register(app)
meta_cmd.register(app)
