"""Validation and mapping helpers for the ImageNatsAdapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "MAX_IMAGE_DIMENSION",
    "MAX_STEPS",
    "ALLOWED_LORA_DIRS",
    "ALLOWED_EMBEDDING_DIRS",
    "_validate_path",
    "_validate_request",
    "_resolve_loras",
    "_map_exception_to_error",
]

# Bounds validation constants
MAX_IMAGE_DIMENSION = 4096
MAX_STEPS = 200

# Allowlisted directories for LoRA and embedding paths
# These are the standard ComfyUI model directories
ALLOWED_LORA_DIRS = [
    Path.home() / "ComfyUI" / "models" / "loras",
    Path.home() / "ComfyUI" / "models" / "lora",
]
ALLOWED_EMBEDDING_DIRS = [
    Path.home() / "ComfyUI" / "models" / "embeddings",
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
    from imagecli.engine import InsufficientResourcesError

    if isinstance(exc, InsufficientResourcesError):
        return "insufficient_resources", "Not enough VRAM or RAM to load engine"
    if isinstance(exc, ValueError) and "Unknown engine" in str(exc):
        return "unknown_engine", str(exc).split(": ", 1)[-1] if ": " in str(exc) else "unknown"
    if isinstance(exc, MemoryError):
        return "insufficient_resources", "Out of memory during generation"

    # Generic fallback - don't leak internal details
    return "generation_failed", "Generation failed"
