#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ALLOWLIST=(
  "src/imagecli/nats/adapter.py"
  "tests/nats/test_adapter.py"
)

HITS=$(grep -rln -E "lyra\.[a-zA-Z_]+\." src/ tests/ --include='*.py' || true)

VIOLATORS=()
while IFS= read -r file; do
  [ -z "$file" ] && continue
  allowed=0
  for allowed_path in "${ALLOWLIST[@]}"; do
    [ "$file" = "$allowed_path" ] && { allowed=1; break; }
  done
  [ "$allowed" -eq 0 ] && VIOLATORS+=("$file")
done <<< "$HITS"

if [ "${#VIOLATORS[@]}" -gt 0 ]; then
  echo "ERROR: lyra.* subject literal outside designated adapter module:"
  printf '  %s\n' "${VIOLATORS[@]}"
  echo "Add to ALLOWLIST in scripts/check-lyra-literals.sh if this is a new designated module (requires ADR)."
  exit 1
fi
