"""Model registry with LRU eviction for VRAM management.

Singleton cache for ImageEngine instances with automatic eviction of
least-recently-used engines when VRAM is insufficient.

Ported from voiceCLI (ADR-002). Intended for long-running hosts (daemon,
NATS adapter) that serve requests across multiple engines — CLI one-shot
flows don't need it since the process dies after each call.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imagecli.engine import ImageEngine

log = logging.getLogger(__name__)

VRAM_OK_MB = 4096
VRAM_CONSTRAINED_MB = 1024


class InsufficientVRAMError(RuntimeError):
    """Raised when VRAM is insufficient even after evicting all cached engines."""


class ModelRegistry:
    """Thread-safe LRU cache for ImageEngine instances with VRAM-aware eviction."""

    def __init__(self, max_cached: int = 2) -> None:
        if max_cached < 1:
            raise ValueError(f"max_cached must be >= 1, got {max_cached}")
        self._cache: OrderedDict[str, ImageEngine] = OrderedDict()
        self._max_cached = max_cached
        self._lock = threading.Lock()
        self._loading: dict[str, threading.Lock] = {}

    def configure(self, max_cached: int) -> None:
        if max_cached < 1:
            raise ValueError(f"max_cached must be >= 1, got {max_cached}")
        with self._lock:
            self._max_cached = max_cached

    def get(self, name: str) -> ImageEngine:
        """Return engine by name. Loads if not cached. Thread-safe.

        Raises:
            ValueError: Unknown engine name.
            InsufficientVRAMError: Not enough VRAM even after full eviction.
        """
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
            if name not in self._loading:
                self._loading[name] = threading.Lock()
            load_lock = self._loading[name]

        with load_lock:
            with self._lock:
                if name in self._cache:
                    self._cache.move_to_end(name)
                    return self._cache[name]

            from imagecli.engine import _get_registry

            engines = _get_registry()
            if name not in engines:
                raise ValueError(f"Unknown engine '{name}'. Available: {list(engines.keys())}")

            # Evict first so there's room for the incoming engine.
            self._ensure_vram(name)

            # Enforce LRU cap — evict oldest if we're at capacity.
            self._enforce_cap()

            engine = engines[name]()

            with self._lock:
                if name not in self._cache:
                    self._cache[name] = engine
                    self._loading.pop(name, None)
                self._cache.move_to_end(name)

            return engine

    def loaded_engines(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def evict(self, name: str) -> None:
        """Evict a specific engine. No-op if not cached."""
        with self._lock:
            if name not in self._cache:
                return
            engine = self._cache.pop(name)
        self._release_engine(engine)

    def evict_all(self) -> None:
        """Evict every cached engine."""
        with self._lock:
            items = list(self._cache.values())
            self._cache.clear()
        for engine in items:
            self._release_engine(engine)

    def vram_free_mb(self) -> int:
        try:
            import torch

            if not torch.cuda.is_available():
                return 0
            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes // (1024 * 1024)
        except Exception as e:
            log.debug("VRAM check failed: %s", e)
            return 0

    def vram_status(self) -> str:
        """Return 'ok' (>4GB), 'constrained' (1-4GB), 'critical' (<1GB)."""
        free_mb = self.vram_free_mb()
        if free_mb >= VRAM_OK_MB:
            return "ok"
        if free_mb >= VRAM_CONSTRAINED_MB:
            return "constrained"
        return "critical"

    # ── Internal ──────────────────────────────────────────────────────────

    def _enforce_cap(self) -> None:
        """Evict LRU engines while cache exceeds max_cached."""
        while True:
            with self._lock:
                if len(self._cache) < self._max_cached:
                    return
                _, engine = self._cache.popitem(last=False)
            self._release_engine(engine)

    def _ensure_vram(self, required: str) -> None:
        """Evict LRU engines until VRAM for `required` is available."""
        from imagecli.engine import _get_registry

        cls = _get_registry().get(required)
        required_gb = float(getattr(cls, "vram_gb", 0.0))

        # CUDA unavailable / unknown size → skip (let preflight fail later).
        if self.vram_free_mb() == 0 or required_gb <= 0:
            return

        while not self._has_vram(required_gb):
            with self._lock:
                if not self._cache:
                    raise InsufficientVRAMError(
                        f"Not enough VRAM for '{required}' even after evicting all "
                        f"engines. Need {required_gb:.1f} GB."
                    )
                _, engine = self._cache.popitem(last=False)
            self._release_engine(engine)

    def _has_vram(self, required_gb: float) -> bool:
        return self.vram_free_mb() >= required_gb * 1024

    def _release_engine(self, engine: ImageEngine) -> None:
        """Release engine's VRAM via its own cleanup() path + forced gc."""
        import gc

        try:
            engine.cleanup()
        except Exception as e:
            log.debug("engine.cleanup() raised: %s", e)
        del engine
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# Singleton instance
model_registry = ModelRegistry()
