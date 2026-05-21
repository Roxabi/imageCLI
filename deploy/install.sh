#!/usr/bin/env bash
# deploy/install.sh — idempotent Quadlet install for imageCLI (image-worker role)
#
# Usage:
#   bash deploy/install.sh [--dry-run] [--secrets-only] [--force]
#
# Flags:
#   --dry-run      Print actions without executing them
#   --secrets-only Only generate secrets (skip Quadlet copy + daemon-reload)
#   --force        Overwrite existing Quadlet files even if unchanged
#
# Prerequisites:
#   - podman (rootless)
#   - systemd --user (lingering enabled for the user)
#
# Standards: S5 (~/.roxabi/imagecli/), S6 (env files in ~/...roxabi/imagecli/env/),
#            S7 (/run/secrets/*.seed), S8 (imagecli-nats-gen), S10 (UID 1503)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="imagecli"
DATA_DIR="${HOME}/.roxabi/${TOOL}"
QUADLET_DIR="${HOME}/.config/containers/systemd"
SECRET_NAME="imagecli-nats-gen"

DRY_RUN=false
SECRETS_ONLY=false
FORCE=false

# ── arg parsing ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --secrets-only) SECRETS_ONLY=true ;;
        --force)        FORCE=true ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
run() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

info() { echo "  $*"; }
ok()   { echo "  ok  $*"; }
skip() { echo "  skip  $*"; }

# ── data dirs ─────────────────────────────────────────────────────────────────
info "Ensuring data directories..."
run mkdir -p \
    "${DATA_DIR}/out" \
    "${DATA_DIR}/nats_out" \
    "${DATA_DIR}/weights" \
    "${DATA_DIR}/env"
ok "Data dirs: ${DATA_DIR}/"

# ── secrets ───────────────────────────────────────────────────────────────────
info "Checking secret: ${SECRET_NAME}..."
if podman secret inspect "${SECRET_NAME}" &>/dev/null; then
    ok "Secret ${SECRET_NAME} already exists"
elif $DRY_RUN; then
    echo "[dry-run] Would prompt to create secret: ${SECRET_NAME}"
else
    echo "Secret '${SECRET_NAME}' not found."
    echo "Options:"
    echo "  1. Paste NKey seed (nk... prefix):"
    echo "     echo 'SUANKEY...' | podman secret create ${SECRET_NAME} -"
    echo "  2. Load from file:"
    echo "     podman secret create ${SECRET_NAME} /path/to/seed.nk"
    echo ""
    echo "Create the secret then re-run this script."
    exit 1
fi

# ── env file ──────────────────────────────────────────────────────────────────
ENV_FILE="${DATA_DIR}/env/gen.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    info "Creating default env file: ${ENV_FILE}"
    run bash -c "cat > '${ENV_FILE}'" <<'EOF'
# imagecli gen worker env (S6: ~/.roxabi/imagecli/env/gen.env)
# HuggingFace cache — shared with other CLIs
HF_HOME=/home/imagecli/.cache/huggingface
# Log verbosity
LOG_LEVEL=INFO
EOF
    ok "Env file created: ${ENV_FILE}"
else
    skip "Env file already exists: ${ENV_FILE}"
fi

# ── quadlet files ─────────────────────────────────────────────────────────────
if $SECRETS_ONLY; then
    info "Skipping Quadlet install (--secrets-only)"
    exit 0
fi

info "Installing Quadlet files to ${QUADLET_DIR}/..."
run mkdir -p "${QUADLET_DIR}"

SRC_CONTAINER="${SCRIPT_DIR}/quadlet/imagecli-gen.container"
DST_CONTAINER="${QUADLET_DIR}/imagecli-gen.container"

if [[ ! -f "${SRC_CONTAINER}" ]]; then
    echo "Source not found: ${SRC_CONTAINER}" >&2
    exit 1
fi

if $FORCE || ! diff -q "${SRC_CONTAINER}" "${DST_CONTAINER}" &>/dev/null; then
    run cp "${SRC_CONTAINER}" "${DST_CONTAINER}"
    ok "Copied imagecli-gen.container"
else
    skip "imagecli-gen.container unchanged"
fi

# ── daemon-reload ─────────────────────────────────────────────────────────────
info "Running systemctl --user daemon-reload..."
run systemctl --user daemon-reload
ok "daemon-reload done"

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Install complete. To start:"
echo "  systemctl --user start imagecli-gen.service"
echo ""
echo "To check status:"
echo "  systemctl --user status imagecli-gen.service"
echo "  journalctl --user -u imagecli-gen.service -f"
echo ""
echo "NOTE: Host data move required before imagecli-gen will load PuLID weights:"
echo "  mv ~/ComfyUI/models/pulid ~/.roxabi/imagecli/weights/pulid"
echo "  mv ~/ComfyUI/models/insightface ~/.roxabi/imagecli/weights/insightface"
echo "  (Phase 1D operator action)"
