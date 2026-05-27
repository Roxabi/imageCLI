"""Validation and mapping helpers for the ImageNatsAdapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from imagecli.engine import InsufficientResourcesError, UnknownEngineError

__all__ = [
    "MAX_IMAGE_DIMENSION",
    "MAX_STEPS",
    "ALLOWED_LORA_DIRS",
    "ALLOWED_EMBEDDING_DIRS",
    "ALLOWED_FORMATS",
    "REQUEST_ID_PATTERN",
    "_validate_path",
    "_validate_request",
    "_resolve_loras",
    "_map_exception_to_error",
    "_sanitize_delivery_exception",
]

# Bounds validation constants
MAX_IMAGE_DIMENSION = 4096
MAX_STEPS = 200

# Filesystem-safe character class for request_id — `request_id` is passed as
# the `filename` field on `BlobRef` (and historically as a path component in
# the now-removed nats_output_dir helper). Keep the charset pinned so future
# downstream consumers (filenames on disk, HTTP headers, etc.) get a safe
# input. Aligns with the contract's `Annotated[str, StringConstraints(min_length=1)]`.
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Allowlisted output formats — matches ImageRequest.format Literal in
# roxabi_contracts.image.models. Same defence-in-depth as request_id: fmt
# reaches the `filename` field on BlobRef and downstream MIME mapping.
ALLOWED_FORMATS = frozenset({"png", "jpeg", "webp"})

# Allowlisted directories for LoRA and embedding paths
# Standard Roxabi data convention: ~/.roxabi/imagecli/ (S5 data dirs standard)
ALLOWED_LORA_DIRS = [
    Path.home() / ".roxabi" / "imagecli" / "loras",
]
ALLOWED_EMBEDDING_DIRS = [
    Path.home() / ".roxabi" / "imagecli" / "embeddings",
]


def _validate_path(path_str: str | None, allowed_dirs: list[Path]) -> tuple[bool, str | None]:
    """Validate that a path is within an allowlisted directory.

    Returns (valid, error_message). Path must be absolute and resolve to a subdir
    of one of the allowed directories.
    """
    if not path_str:
        return True, None

    path = Path(path_str)

    # Must be absolute
    if not path.is_absolute():
        return False, f"path must be absolute: {path_str}"

    # Resolve to canonical form (resolves .. and symlinks)
    try:
        resolved = path.resolve()
    except OSError as e:
        return False, f"invalid path: {e}"

    # Check against allowed directories
    for allowed_dir in allowed_dirs:
        try:
            resolved.relative_to(allowed_dir.resolve())
            return True, None  # Path is under this allowed directory
        except ValueError:
            continue

    allowed_str = ", ".join(str(d) for d in allowed_dirs)
    return False, f"path not in allowed directories: {path_str} (allowed: {allowed_str})"


def _validate_request(payload: dict) -> tuple[bool, str | None]:
    """Validate required fields and bounds. Returns (valid, error_message)."""
    if not payload.get("prompt"):
        return False, "missing_required_field: prompt"
    if not payload.get("engine"):
        return False, "missing_required_field: engine"

    # request_id reaches the filesystem via `nats_{request_id[:8]}.{fmt}` —
    # path-traversal sink. Allowlist filesystem-safe chars.
    request_id = payload.get("request_id", "")
    if not REQUEST_ID_PATTERN.match(request_id):
        return False, "request_id must match [A-Za-z0-9_-]{1,128}"

    # format also reaches the filesystem (filename suffix + tempfile suffix).
    fmt = payload.get("format")
    if fmt is not None and fmt not in ALLOWED_FORMATS:
        return False, f"format must be one of {sorted(ALLOWED_FORMATS)}: got {fmt!r}"

    # Bounds validation for dimensions
    width = payload.get("width")
    height = payload.get("height")
    steps = payload.get("steps")

    if width is not None and width > MAX_IMAGE_DIMENSION:
        return False, f"width exceeds max ({width} > {MAX_IMAGE_DIMENSION})"
    if height is not None and height > MAX_IMAGE_DIMENSION:
        return False, f"height exceeds max ({height} > {MAX_IMAGE_DIMENSION})"
    if steps is not None and steps > MAX_STEPS:
        return False, f"steps exceeds max ({steps} > {MAX_STEPS})"

    # Mixed-form guard: reject payloads that combine loras list with singular keys
    _SINGULAR_LORA_KEYS = {"lora_path", "lora_scale", "trigger", "embedding_path"}
    has_loras_list = "loras" in payload
    singular_present = _SINGULAR_LORA_KEYS & payload.keys()
    if has_loras_list and singular_present:
        return (
            False,
            "Pass either loras= or the singular fields "
            "(lora_path / lora_scale / trigger / embedding_path), not both.",
        )

    # Path validation for singular lora_path / embedding_path
    if not has_loras_list:
        lora_path = payload.get("lora_path")
        embedding_path = payload.get("embedding_path")

        valid, err = _validate_path(lora_path, ALLOWED_LORA_DIRS)
        if not valid:
            return False, f"lora_path: {err}"

        valid, err = _validate_path(embedding_path, ALLOWED_EMBEDDING_DIRS)
        if not valid:
            return False, f"embedding_path: {err}"
    else:
        from imagecli.lora_spec import MAX_LORAS

        loras_list = payload.get("loras") or []
        if len(loras_list) > MAX_LORAS:
            return (
                False,
                f"loras list exceeds cap: {len(loras_list)} > {MAX_LORAS}",
            )
        # Validate each lora entry's path and embedding_path
        for i, item in enumerate(loras_list):
            if not isinstance(item, dict):
                return False, f"loras[{i}] must be a mapping, got {type(item).__name__}"
            if "path" not in item:
                return False, f"loras[{i}] missing required key 'path'"
            valid, err = _validate_path(item.get("path"), ALLOWED_LORA_DIRS)
            if not valid:
                return False, f"loras[{i}].path: {err}"
            emb = item.get("embedding_path")
            if emb:
                valid, err = _validate_path(emb, ALLOWED_EMBEDDING_DIRS)
                if not valid:
                    return False, f"loras[{i}].embedding_path: {err}"

    return True, None


def _resolve_loras(payload: dict) -> list[Any]:
    """Resolve lora specs from payload (list form or singular keys).

    Returns a list of LoraSpec objects. Mixed-form payloads must already be
    rejected by _validate_request before calling this function.
    """
    from imagecli.lora_spec import LoraSpec

    raw_loras = payload.get("loras")
    if raw_loras is not None:
        return [
            LoraSpec(
                path=str(item["path"]),
                scale=float(item.get("scale", 1.0)),
                trigger=item.get("trigger") or None,
                embedding_path=item.get("embedding_path") or None,
            )
            for item in raw_loras
        ]

    lora_path = payload.get("lora_path")
    if lora_path:
        return [
            LoraSpec(
                path=lora_path,
                scale=float(payload.get("lora_scale", 1.0)),
                trigger=payload.get("trigger"),
                embedding_path=payload.get("embedding_path"),
            )
        ]
    return []


def _map_exception_to_error(exc: Exception) -> tuple[str, str]:
    """Map exception types to error codes and sanitized messages.

    Returns (error_code, error_detail). The error_detail is sanitized
    to avoid leaking paths, usernames, or internal structure.
    """
    if isinstance(exc, InsufficientResourcesError):
        return "insufficient_resources", "Not enough VRAM or RAM to load engine"
    if isinstance(exc, UnknownEngineError):
        detail = ", ".join(exc.available) if exc.available else "unknown"
        return "unknown_engine", detail
    if isinstance(exc, MemoryError):
        return "insufficient_resources", "Out of memory during generation"

    # Generic fallback - don't leak internal details
    return "generation_failed", "Generation failed"


def _sanitize_delivery_exception(exc: Exception) -> str:
    """Map an httpx / BlobStore exception to a safe diagnostic string.

    Sibling to :func:`_map_exception_to_error` — engine-side errors stay there,
    delivery-side (HttpBlobStore) errors live here. The two stay separate
    because their type domains are orthogonal (``httpx.*`` vs
    ``imagecli.engine.*``).

    The returned string is wire-facing (``WorkerError.detail``) AND log-facing
    (``_probe_blobstore`` warning), so it MUST NOT carry URLs, headers, or any
    string serialised from ``httpx.Request`` — those can embed the Bearer
    token or query-string auth. We classify by ``isinstance`` only and never
    interpolate ``exc`` directly.
    """
    # httpx is an optional transitive dep of roxabi-blobs.HttpBlobStore — lazy
    # import so this helper stays usable even when httpx isn't installed
    # (e.g., the legacy CLI path that never exercises the NATS adapter).
    try:
        import httpx
    except ImportError:
        return "internal delivery error"

    if isinstance(exc, httpx.HTTPStatusError):
        # Narrowed by isinstance at runtime; httpx isn't statically typed here
        # because of the lazy import.
        status_code = exc.response.status_code  # type: ignore[attr-defined]
        return f"upstream HTTP {status_code}"
    if isinstance(exc, httpx.TimeoutException):
        return "BlobStore request timed out"
    if isinstance(exc, httpx.ConnectError):
        return "BlobStore connection failed"
    if isinstance(exc, httpx.HTTPError):
        # Generic httpx parent — covers ProtocolError, RemoteProtocolError, etc.
        return "BlobStore transport error"
    return "internal delivery error"
