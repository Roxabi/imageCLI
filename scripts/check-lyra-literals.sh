#!/usr/bin/env bash
set -euo pipefail
ALLOWLIST="src/imagecli/nats/adapter.py"
VIOLATORS=$(grep -rln "lyra\." src/ --include='*.py' | grep -vE "^($(echo "$ALLOWLIST" | tr ' ' '|'))$" || true)
if [ -n "$VIOLATORS" ]; then
  echo "ERROR: lyra.* subject literal outside designated adapter module:"
  echo "$VIOLATORS"
  echo "Add to allowlist in scripts/check-lyra-literals.sh if this is a new designated module (requires ADR)."
  exit 1
fi
