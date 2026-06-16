#!/usr/bin/env bash
# check_healthcmd_quoting.sh — scan Quadlet unit files for the trailing-double-quote
# HealthCmd bug (Podman ≤5.7.0 strips a trailing " from HealthCmd values, producing an
# unterminated shell string and false-"unhealthy" status while the service runs fine).
#
# Exit codes (dev-core contract):
#   0 — all HealthCmd values clean
#   1 — one or more violations found (print file:line for each)
#   2 — script error (bad environment, missing required tools, etc.)
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "ERROR: not inside a git repository" >&2
    exit 2
}

# Scan deploy/quadlet/*.container and *.container.disabled
SEARCH_DIR="$REPO_ROOT/deploy/quadlet"

if [ ! -d "$SEARCH_DIR" ]; then
    echo "WARN: $SEARCH_DIR not found — no Quadlet units to scan" >&2
    exit 0
fi

FAIL=0

# Find all .container and .container.disabled files
while IFS= read -r -d '' unit_file; do
    rel="${unit_file#"$REPO_ROOT/"}"
    lineno=0
    while IFS= read -r line; do
        lineno=$((lineno + 1))
        # Match HealthCmd= lines
        if [[ "$line" =~ ^[[:space:]]*HealthCmd= ]]; then
            # Strip trailing whitespace to get the last non-whitespace character
            trimmed="${line%"${line##*[! ]}"}"
            last_char="${trimmed: -1}"
            if [ "$last_char" = '"' ]; then
                echo "FAIL: $rel:$lineno — HealthCmd ends with '\"' (Podman ≤5.7.0 strips trailing quote → false-unhealthy)"
                echo "      $line"
                FAIL=1
            fi
        fi
    done < "$unit_file"
done < <(find "$SEARCH_DIR" \( -name "*.container" -o -name "*.container.disabled" \) -print0)

if [ "$FAIL" -eq 0 ]; then
    echo "OK: no trailing-quote HealthCmd found in $SEARCH_DIR"
fi

exit $FAIL
