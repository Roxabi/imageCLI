# Configuration

`imagecli.toml` searched CWD → `$HOME`. Global: `~/imagecli.toml`.
Priority: CLI flag > frontmatter > imagecli.toml > default.

## Blobstore (NATS worker only, #97)

`[blobstore]` stanza: `endpoint` + `token` for the cross-host HttpBlobStore (M₂ → M₁ lyra-blobstore on port 8449).

Server-config precedence:
1. `imagecli.toml [blobstore]`
2. env (`IMAGECLI_BLOBSTORE_URL`, `IMAGECLI_BLOBSTORE_TOKEN`)
3. default (`http://roxabituwer:8449`, no token)

Missing token → fail-fast at adapter init. Production: token via Quadlet secret `imagecli-blobstore-token`.
