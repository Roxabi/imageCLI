"""HttpBlobStore singleton for imageCLI NATS worker — mirrors voiceCLI pattern."""

from __future__ import annotations

import threading

from roxabi_blobs.http_store import HttpBlobStore

_INSTANCE: HttpBlobStore | None = None
_LOCK = threading.Lock()


def get_blobstore() -> HttpBlobStore:
    """Return process-local HttpBlobStore, creating from config on first call."""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                from imagecli.config import load_blobstore_config

                cfg = load_blobstore_config()
                if cfg["token"] is None:
                    raise RuntimeError(
                        "blobstore token not configured — set IMAGECLI_BLOBSTORE_TOKEN_PATH, "
                        "imagecli.toml [blobstore].token, or IMAGECLI_BLOBSTORE_TOKEN"
                    )
                _INSTANCE = HttpBlobStore(base_url=cfg["endpoint"], token=cfg["token"])
    return _INSTANCE


def reset_blobstore_for_tests() -> None:
    """Reset singleton between tests."""
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None