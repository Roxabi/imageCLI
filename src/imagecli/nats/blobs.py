"""HttpBlobStore singleton for imageCLI NATS worker — mirrors voiceCLI pattern."""

from __future__ import annotations

import threading

from roxabi_blobs.http_store import HttpBlobStore

_blobstore_instance: HttpBlobStore | None = None
_LOCK = threading.Lock()


def get_blobstore() -> HttpBlobStore:
    """Return process-local HttpBlobStore, creating from config on first call."""
    global _blobstore_instance
    if _blobstore_instance is None:
        with _LOCK:
            if _blobstore_instance is None:
                from imagecli.config import load_blobstore_config

                cfg = load_blobstore_config()
                if cfg["token"] is None:
                    raise RuntimeError(
                        "blobstore token not configured — set IMAGECLI_BLOBSTORE_TOKEN_PATH, "
                        "imagecli.toml [blobstore].token, or IMAGECLI_BLOBSTORE_TOKEN"
                    )
                _blobstore_instance = HttpBlobStore(
                    base_url=cfg["endpoint"], token=cfg["token"]
                )
    return _blobstore_instance


def reset_blobstore_for_tests() -> None:
    """Reset singleton between tests."""
    global _blobstore_instance
    with _LOCK:
        _blobstore_instance = None