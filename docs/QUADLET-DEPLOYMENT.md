# imageCLI — Quadlet Deployment Runbook

Deployment via Podman Quadlet (systemd `--user`). Host role: `image-worker` (M₂ only).

→ Standard: `~/projects/docs/container-deployment-standard.md`

## Prerequisites

- Host: `roxabitower` (M₂, RTX 5070 Ti, `image-worker` role)
- Podman rootless + `loginctl enable-linger $(whoami)`
- Secret `imagecli-nats-gen` must exist (NATS NKey seed)
- Data move done (Phase 1D): `~/ComfyUI/models/pulid` → `~/.roxabi/imagecli/weights/pulid`

## Install (idempotent)

```bash
cd ~/projects/imageCLI
bash deploy/install.sh
```

Flags: `--dry-run` | `--secrets-only` | `--force`

## Secret Management

### Create secret (first install)

```bash
# From NKey file
podman secret create imagecli-nats-gen /path/to/nkey.seed

# From stdin
echo 'SUANKEY...' | podman secret create imagecli-nats-gen -
```

### Rotate secret (atomique)

```bash
# 1. Create new secret
podman secret inspect --showsecret imagecli-nats-gen | jq -r '.[0].SecretData' \
  | podman secret create imagecli-nats-gen-new -

# 2. Update Quadlet (if name changes)
sed -i 's/imagecli-nats-gen/imagecli-nats-gen-new/' \
  ~/.config/containers/systemd/imagecli-gen.container

# 3. Reload + restart
systemctl --user daemon-reload
systemctl --user restart imagecli-gen.service

# 4. Remove old secret
podman secret rm imagecli-nats-gen
```

## Service Management

```bash
# Start
systemctl --user start imagecli-gen.service

# Stop
systemctl --user stop imagecli-gen.service

# Status
systemctl --user status imagecli-gen.service
podman ps --filter name=imagecli-gen

# Logs
journalctl --user -u imagecli-gen.service -f

# Restart
systemctl --user restart imagecli-gen.service
```

## Data Directories

| Path | Purpose |
|---|---|
| `~/.roxabi/imagecli/out/` | CLI output images |
| `~/.roxabi/imagecli/nats_out/` | NATS satellite output images |
| `~/.roxabi/imagecli/weights/` | PuLID + InsightFace weights (read-only in container) |
| `~/.roxabi/imagecli/env/gen.env` | Runtime env (HF_HOME, LOG_LEVEL) |
| `~/.cache/huggingface/` | HuggingFace model cache (shared with other CLIs) |

## Volumes (container-side)

| Mount | Container path | Mode |
|---|---|---|
| `~/.cache/huggingface` | `/home/imagecli/.cache/huggingface` | rw |
| `~/.roxabi/imagecli` | `/home/imagecli/.roxabi/imagecli` | rw |
| `~/.roxabi/imagecli/weights` | `/home/imagecli/.roxabi/imagecli/weights` | ro,Z |

## Diagnostics

```bash
# Check NATS connectivity
podman exec imagecli-gen nats pub test.ping ""

# Check GPU access
podman exec imagecli-gen nvidia-smi

# Check weights mount
podman exec imagecli-gen ls /home/imagecli/.roxabi/imagecli/weights/pulid/

# Check secret mount
podman exec imagecli-gen ls /run/secrets/
```

## Auto-update

Image auto-updates via `podman-auto-update.timer` (label `io.containers.autoupdate=registry`).

```bash
# Force update check
podman auto-update --dry-run

# Status
systemctl --user status podman-auto-update.timer
```

## Phase 1D Operator Actions (REQUIRED before PuLID works)

```bash
# Move PuLID weights
mkdir -p ~/.roxabi/imagecli/weights
mv ~/ComfyUI/models/pulid ~/.roxabi/imagecli/weights/pulid
mv ~/ComfyUI/models/insightface ~/.roxabi/imagecli/weights/insightface
# Verify hierarchy:
ls ~/.roxabi/imagecli/weights/pulid/
ls ~/.roxabi/imagecli/weights/insightface/models/
```

These moves are NOT automated — they are operator responsibility (Phase 1D).
