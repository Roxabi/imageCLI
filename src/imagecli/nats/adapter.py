"""ImageNatsAdapter — imageCLI NATS satellite for async image generation requests."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

from roxabi_nats import NatsAdapterBase

log = logging.getLogger(__name__)

SUBJECT = "lyra.image.generate.request"
HEARTBEAT_SUBJECT = "lyra.image.heartbeat"
SCHEMA_VERSION = "1"

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


class ImageNatsAdapter(NatsAdapterBase):
    """NATS adapter for image generation requests from Lyra hub.

    Subscribes to `lyra.image.generate.request` and responds with generated images.
    Follows the satellite-bootstrap pattern established by voice adapters (ADR-039).
    """

    def __init__(
        self,
        default_engine: str = "flux2-klein",
        *,
        max_concurrent: int = 1,
        heartbeat_interval: float = 5.0,
        drain_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            subject=SUBJECT,
            queue_group="IMAGE_WORKERS",
            envelope_name="image",
            schema_version=SCHEMA_VERSION,
            heartbeat_subject=HEARTBEAT_SUBJECT,
            heartbeat_interval=heartbeat_interval,
            drain_timeout=drain_timeout,
        )
        self.default_engine = default_engine
        self.max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._engine_loaded: str | None = None
        self._engine_instance: Any = None  # Cached engine for reuse

    def _validate_path(
        self, path_str: str | None, allowed_dirs: list[Path]
    ) -> tuple[bool, str | None]:
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

    def _validate_request(self, payload: dict) -> tuple[bool, str | None]:
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

        # Path validation for LoRA and embeddings
        lora_path = payload.get("lora_path")
        embedding_path = payload.get("embedding_path")

        valid, err = self._validate_path(lora_path, ALLOWED_LORA_DIRS)
        if not valid:
            return False, f"lora_path: {err}"

        valid, err = self._validate_path(embedding_path, ALLOWED_EMBEDDING_DIRS)
        if not valid:
            return False, f"embedding_path: {err}"

        return True, None

    def _map_exception_to_error(self, exc: Exception) -> tuple[str, str]:
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

    async def handle(self, msg: Any, payload: dict) -> None:
        """Process an image generation request per ADR-046 contract."""
        request_id = payload.get("request_id", "")

        # Acquire semaphore for concurrency control
        async with self._sem:
            # Validate required fields and bounds
            valid, error = self._validate_request(payload)
            if not valid:
                await self._reply_error(msg, request_id, "missing_required_field", error)
                return

            engine_name = payload["engine"]
            prompt = payload["prompt"]

            # Import engine module lazily (heavy torch imports)
            try:
                from imagecli.engine import get_engine, preflight_check, list_engines
            except ImportError as e:
                await self._reply_error(msg, request_id, "engine_load_failed", f"Import error: {e}")
                return

            # Validate engine name against registry
            engine_names = {e["name"] for e in list_engines()}
            if engine_name not in engine_names:
                await self._reply_error(msg, request_id, "unknown_engine", engine_name)
                return

            # Get or create engine instance.
            # No-LoRA / no-pivotal requests share a cached instance via
            # ModelRegistry (warm across requests, LRU-evicted on VRAM
            # pressure). Per-request LoRA/pivotal configs take a fresh
            # engine (registry keys by name only) and cleanup() after use.
            lora_path = payload.get("lora_path")
            lora_scale = payload.get("lora_scale", 1.0)
            trigger = payload.get("trigger")
            embedding_path = payload.get("embedding_path")
            uses_per_request_cfg = bool(lora_path or trigger or embedding_path)

            try:
                if uses_per_request_cfg:
                    engine = get_engine(
                        engine_name,
                        compile=True,
                        lora_path=lora_path,
                        lora_scale=lora_scale,
                        trigger=trigger,
                        embedding_path=embedding_path,
                    )
                else:
                    from imagecli.model_registry import model_registry

                    engine = model_registry.get(engine_name)
                self._engine_loaded = engine_name
            except Exception as e:
                error_code, error_detail = self._map_exception_to_error(e)
                await self._reply_error(msg, request_id, error_code, error_detail)
                return

            # Preflight check (VRAM/RAM validation)
            try:
                preflight_check(engine)
            except Exception as e:
                error_code, error_detail = self._map_exception_to_error(e)
                await self._reply_error(msg, request_id, error_code, error_detail)
                return

            # Extract generation params with defaults
            width = payload.get("width", 1024)
            height = payload.get("height", 1024)
            steps = payload.get("steps", 50)
            guidance = payload.get("guidance", 4.0)
            seed = payload.get("seed")
            negative_prompt = payload.get("negative_prompt", "")
            fmt = payload.get("format", "png")
            output_mode = payload.get("output_mode", "b64")

            # Generate image
            start_time = time.monotonic()
            try:
                # Generate to a temp file first
                from tempfile import NamedTemporaryFile

                with NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                saved_path = engine.generate(
                    prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    output_path=tmp_path,
                )

                duration = time.monotonic() - start_time
                log.info(f"Generated {saved_path} in {duration:.2f}s with engine {engine_name}")

            except Exception as e:
                error_code, error_detail = self._map_exception_to_error(e)
                log.exception(f"Generation failed: {e}")
                if uses_per_request_cfg:
                    engine.cleanup()
                else:
                    from imagecli.model_registry import model_registry

                    # Evict so the registry can reload cleanly next request.
                    model_registry.evict(engine_name)
                await self._reply_error(msg, request_id, error_code, error_detail)
                return

            # Read and encode image
            try:
                image_bytes = saved_path.read_bytes()

                if output_mode == "b64":
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                    # Check if base64 exceeds reasonable NATS payload limit (~75% of 1MB)
                    max_b64_size = 750_000  # ~750KB
                    if len(image_b64) > max_b64_size:
                        # Fall back to file mode - move to images_out
                        output_dir = Path("images/images_out")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        final_path = output_dir / f"nats_{request_id[:8]}.{fmt}"
                        saved_path.rename(final_path)

                        reply = {
                            "contract_version": "1",
                            "request_id": request_id,
                            "ok": True,
                            "file_path": str(final_path.resolve()),
                            "mime_type": f"image/{fmt}",
                            "width": width,
                            "height": height,
                            "engine": engine_name,
                            "seed_used": seed if seed else 0,  # Would need to extract from engine
                        }
                    else:
                        reply = {
                            "contract_version": "1",
                            "request_id": request_id,
                            "ok": True,
                            "image_b64": image_b64,
                            "mime_type": f"image/{fmt}",
                            "width": width,
                            "height": height,
                            "engine": engine_name,
                            "seed_used": seed if seed else 0,
                        }
                else:  # output_mode == "file"
                    # Move to images_out
                    output_dir = Path("images/images_out")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    final_path = output_dir / f"nats_{request_id[:8]}.{fmt}"
                    saved_path.rename(final_path)

                    reply = {
                        "contract_version": "1",
                        "request_id": request_id,
                        "ok": True,
                        "file_path": str(final_path.resolve()),
                        "mime_type": f"image/{fmt}",
                        "width": width,
                        "height": height,
                        "engine": engine_name,
                        "seed_used": seed if seed else 0,
                    }

                await self.reply(msg, json.dumps(reply).encode())

            except Exception as e:
                log.exception(f"Failed to encode response: {e}")
                await self._reply_error(
                    msg, request_id, "generation_failed", "Failed to encode response"
                )
            finally:
                if uses_per_request_cfg:
                    engine.cleanup()
                else:
                    # Registry-cached engine stays warm between requests.
                    engine.clear_cache()
                # Clean up temp file if it still exists
                try:
                    if saved_path.exists():
                        saved_path.unlink()
                except OSError:
                    pass

    async def _reply_error(
        self, msg: Any, request_id: str, error: str, error_detail: str | None = None
    ) -> None:
        """Send an error reply following ADR-046 contract."""
        reply = {
            "contract_version": "1",
            "request_id": request_id,
            "ok": False,
            "error": error,
        }
        if error_detail:
            reply["error_detail"] = error_detail
        await self.reply(msg, json.dumps(reply).encode())

    def heartbeat_payload(self) -> dict:
        """Extend base heartbeat per lyra.image contract (see roxabi_contracts.image.models.Heartbeat)."""
        base = super().heartbeat_payload()
        from imagecli.model_registry import model_registry

        loaded = model_registry.loaded_engines()
        # Contract exposes a single engine_loaded string; pick the MRU entry
        # when the registry caches several, else fall back to per-request tracking.
        base["engine_loaded"] = loaded[-1] if loaded else self._engine_loaded
        base["active_requests"] = self._active_request_count()

        free_mb = model_registry.vram_free_mb()
        total_mb = 0
        try:
            import torch

            if torch.cuda.is_available():
                total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        except Exception:
            pass
        if total_mb:
            base["vram_used_mb"] = int(total_mb - free_mb)
            base["vram_total_mb"] = int(total_mb)
        return base

    def _active_request_count(self) -> int:
        """Return current active request count for heartbeat."""
        # Semaphore-based counting: unavailable permits = active requests
        # max_concurrent - _sem._value gives in-flight count
        return self.max_concurrent - self._sem._value
