# imageCLI — Quadlet Deployment Runbook

Deployment via Podman Quadlet (systemd `--user`). Host role: `image-worker` (M₂ only).

→ Standard: `~/projects/docs/container-deployment-standard.md`

## Prerequisites

- Host: `roxabitower` (M₂, RTX 5070 Ti, `image-worker` role)
- Podman rootless + `loginctl enable-linger $(whoami)`
- Secret `imagecli-nats-gen` must exist (NATS NKey seed)
- Secret `imagecli-blobstore-token` must exist (Bearer token for lyra-blobstore HTTP service — see [Blobstore token secret](#blobstore-token-secret-97) below)
- Data move done (Phase 1D): `~/ComfyUI/models/pulid` → `~/.roxabi/imagecli/weights/pulid`

## Install (idempotent)

```bash
cd ~/projects/imageCLI
bash deploy/install.sh
```

Flags: `--dry-run` | `--secrets-only` | `--force`

## Canonical nkeys path

Lyra's `acl-matrix.json` (Roxabi/lyra#1381) scp's the regenerated NKey seed for `image-worker` to:

```
~/.roxabi/imagecli/nkeys/image-worker.seed
```

`deploy/install.sh` creates `~/.roxabi/imagecli/nkeys/` via `mkdir -p` and prints the resolved path at install time. Override with `IMAGECLI_SEED_PATH=/custom/path` if needed.

**Operator workflow after `lyra-acl genkeys --regenerate`:**

1. Lyra fires the external scp manifest — seed lands at `~/.roxabi/imagecli/nkeys/image-worker.seed` on `roxabitower`
2. **Secure the seed** (mandatory — scp defaults to 644):
   ```bash
   chmod 400 ~/.roxabi/imagecli/nkeys/image-worker.seed
   ```
3. Recreate the Podman secret from the new seed:
   ```bash
   # Podman 4.3+ — verify with: podman --version
   podman secret create --replace imagecli-nats-gen ~/.roxabi/imagecli/nkeys/image-worker.seed
   ```
4. Restart the service:
   ```bash
   systemctl --user restart imagecli-gen.service
   ```

Refs: Roxabi/lyra#1382, Roxabi/lyra#1381

## Blobstore token secret (#97)

The image-worker PUTs generated images to the cross-host `lyra-blobstore.container` on M₁ (port 8449) using a Bearer token. The token is supplied via the Quadlet secret `imagecli-blobstore-token`, mounted at `/run/secrets/imagecli-blobstore-token` (mount-type, UID 1503 mode 0400).

### First install

1. Obtain the Bearer token from the lyra-blobstore operator.
2. Write it to the canonical token path on M₂:
   ```bash
   mkdir -p ~/.roxabi/imagecli/tokens
   chmod 700 ~/.roxabi/imagecli/tokens
   printf '%s' "$TOKEN" > ~/.roxabi/imagecli/tokens/blobstore.token
   chmod 400 ~/.roxabi/imagecli/tokens/blobstore.token
   ```
3. Run `bash deploy/install.sh` — it will create the `imagecli-blobstore-token` secret from this file.

Override the source path via `IMAGECLI_BLOBSTORE_TOKEN_PATH=/custom/path`.

### Rotation

Mount-type secrets are bound at container init — rotating the secret requires a **container restart**, not just a daemon-reload.

```bash
# 1. Place the new token on disk
printf '%s' "$NEW_TOKEN" > ~/.roxabi/imagecli/tokens/blobstore.token
chmod 400 ~/.roxabi/imagecli/tokens/blobstore.token

# 2. Replace the secret (atomic; Podman 4.3+ — verify with: podman --version)
podman secret create --replace imagecli-blobstore-token ~/.roxabi/imagecli/tokens/blobstore.token

# 3. Restart the service (MANDATORY — mount-type secrets bind at init)
systemctl --user restart imagecli-gen.service
```

Verify the restart succeeded:
```bash
journalctl --user -u imagecli-gen -n 50 | grep -i blobstore
# Expect no HttpBlobStore probe warning, or a successful first put() in subsequent logs.
```

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
