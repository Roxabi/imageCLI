#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ALLOWLIST — edit requires ADR-047 update.
ALLOWLIST=(
  "src/imagecli/nats/adapter.py"
  "tests/nats/test_adapter.py"
)

set +e
HITS=$(grep -rln -E "lyra\.[a-zA-Z_]+\." src/ tests/ --include='*.py')
rc=$?
set -e
# grep exit codes: 0 = matches, 1 = no matches, 2+ = error
if [ "$rc" -ge 2 ]; then
  echo "ERROR: grep failed with exit code $rc (unreadable directory or bad pattern)" >&2
  exit "$rc"
fi

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
