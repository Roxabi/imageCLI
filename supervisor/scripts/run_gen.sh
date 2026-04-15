#!/usr/bin/env bash
# Wrapper for imagecli_gen daemon — sources .env before launching.
# supervisor conf points to this script so secrets never live in conf files.
set -a
[ -f "$HOME/projects/imageCLI/.env" ] && source "$HOME/projects/imageCLI/.env"
set +a
exec imagecli serve --engine flux2-klein
