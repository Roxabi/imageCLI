"""imagecli CLI command modules.

Each submodule defines a Typer command (or group) and exposes ``register(app)``
so ``cli.py`` owns the single Typer ``app`` instance and wires everything.
"""
