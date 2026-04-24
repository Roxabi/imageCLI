"""Tests for imagecli.model_registry — LRU cache + VRAM-aware eviction."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest


class TestCore:
    def test_get_returns_cached_on_second_call(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry()
        mock_engine = MagicMock()

        with (
            patch("imagecli.engine._get_registry") as mock_reg,
            patch.object(registry, "_has_vram", return_value=True),
        ):
            mock_reg.return_value = {"mock": lambda: mock_engine}
            first = registry.get("mock")
            second = registry.get("mock")

        assert first is second

    def test_get_raises_for_unknown_engine(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry()
        with pytest.raises(ValueError, match="Unknown engine"):
            registry.get("nonexistent")

    def test_loaded_engines_lists_cache_in_lru_order(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry(max_cached=3)
        e1, e2 = MagicMock(), MagicMock()

        with (
            patch("imagecli.engine._get_registry") as mock_reg,
            patch.object(registry, "_has_vram", return_value=True),
        ):
            mock_reg.return_value = {"a": lambda: e1, "b": lambda: e2}
            registry.get("a")
            registry.get("b")
            registry.get("a")  # bump a to MRU

        assert registry.loaded_engines() == ["b", "a"]


class TestVRAMEviction:
    def test_evict_removes_engine(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry()
        mock_engine = MagicMock()
        with (
            patch("imagecli.engine._get_registry") as mock_reg,
            patch.object(registry, "_has_vram", return_value=True),
        ):
            mock_reg.return_value = {"mock": lambda: mock_engine}
            registry.get("mock")
        registry.evict("mock")
        assert registry.loaded_engines() == []
        mock_engine.cleanup.assert_called_once()

    def test_evict_noop_for_uncached(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry()
        registry.evict("nonexistent")  # must not raise
        assert registry.loaded_engines() == []

    def test_evict_all_clears_everything(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry(max_cached=3)
        e1, e2 = MagicMock(), MagicMock()
        with (
            patch("imagecli.engine._get_registry") as mock_reg,
            patch.object(registry, "_has_vram", return_value=True),
        ):
            mock_reg.return_value = {"a": lambda: e1, "b": lambda: e2}
            registry.get("a")
            registry.get("b")
        registry.evict_all()
        assert registry.loaded_engines() == []
        e1.cleanup.assert_called_once()
        e2.cleanup.assert_called_once()

    def test_ensure_vram_raises_when_empty(self):
        from imagecli.model_registry import InsufficientVRAMError, ModelRegistry

        registry = ModelRegistry()
        fake_cls = MagicMock(vram_gb=8.0)
        with (
            patch("imagecli.engine._get_registry", return_value={"x": fake_cls}),
            patch.object(registry, "_has_vram", return_value=False),
            patch.object(registry, "vram_free_mb", return_value=1000),
        ):
            with pytest.raises(InsufficientVRAMError, match="Not enough VRAM"):
                registry._ensure_vram("x")

    def test_ensure_vram_evicts_until_sufficient(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry(max_cached=3)
        e1, e2 = MagicMock(), MagicMock()
        fake_cls = MagicMock(vram_gb=8.0)

        with (
            patch("imagecli.engine._get_registry") as mock_reg,
            patch.object(registry, "_has_vram") as mock_has,
            patch.object(registry, "vram_free_mb", return_value=1000),
        ):
            mock_reg.return_value = {
                "a": lambda: e1,
                "b": lambda: e2,
                "c": fake_cls,
            }
            # seed cache with two engines under plenty-VRAM conditions
            mock_has.return_value = True
            registry.get("a")
            registry.get("b")

            # now require c — first check False, eviction, second check True
            mock_has.side_effect = [False, True]
            registry._ensure_vram("c")

        # Oldest ("a") evicted; "b" still cached.
        assert registry.loaded_engines() == ["b"]
        e1.cleanup.assert_called_once()

    def test_cap_evicts_oldest(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry(max_cached=2)
        engines = {f"e{i}": MagicMock() for i in range(3)}
        with (
            patch("imagecli.engine._get_registry") as mock_reg,
            patch.object(registry, "_has_vram", return_value=True),
        ):
            mock_reg.return_value = {k: (lambda v=v: v) for k, v in engines.items()}
            registry.get("e0")
            registry.get("e1")
            registry.get("e2")

        assert registry.loaded_engines() == ["e1", "e2"]
        engines["e0"].cleanup.assert_called_once()


class TestThreadSafety:
    def test_concurrent_get_same_engine_loads_once(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry()
        load_count = 0
        lock = threading.Lock()
        results = []

        def factory():
            nonlocal load_count
            with lock:
                load_count += 1
            return MagicMock()

        with (
            patch("imagecli.engine._get_registry") as mock_reg,
            patch.object(registry, "_has_vram", return_value=True),
        ):
            mock_reg.return_value = {"mock": factory}

            def worker():
                results.append(registry.get("mock"))

            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        assert load_count == 1
        assert results[0] is results[1]


class TestVRAMStatus:
    @pytest.mark.parametrize(
        "free_mb,expected",
        [
            (5000, "ok"),
            (2000, "constrained"),
            (500, "critical"),
        ],
    )
    def test_vram_status(self, free_mb, expected):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry()
        with patch.object(registry, "vram_free_mb", return_value=free_mb):
            assert registry.vram_status() == expected


class TestConfiguration:
    def test_configure_rejects_zero(self):
        from imagecli.model_registry import ModelRegistry

        registry = ModelRegistry()
        with pytest.raises(ValueError):
            registry.configure(0)

    def test_init_rejects_zero(self):
        from imagecli.model_registry import ModelRegistry

        with pytest.raises(ValueError):
            ModelRegistry(max_cached=0)
