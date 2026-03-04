"""Tests for engine.py — registry, defaults, error handling."""

from __future__ import annotations

import pytest

from imagecli.engine import get_engine, get_engine_class, list_engines


def test_list_engines_returns_all_three():
    engines = list_engines()
    names = {e["name"] for e in engines}
    assert names == {"flux2-klein", "flux1-dev", "sd35"}


def test_list_engines_has_required_fields():
    for e in list_engines():
        assert "name" in e
        assert "description" in e
        assert "model_id" in e
        assert "vram_gb" in e
        assert "default_steps" in e
        assert "default_guidance" in e


def test_engine_defaults_per_engine():
    engines = {e["name"]: e for e in list_engines()}
    # flux2-klein: distilled, fewer steps
    assert engines["flux2-klein"]["default_steps"] == 28
    assert engines["flux2-klein"]["default_guidance"] == 3.5
    # flux1-dev: BFL recommendation
    assert engines["flux1-dev"]["default_steps"] == 50
    assert engines["flux1-dev"]["default_guidance"] == 3.5
    # sd35: turbo — fixed
    assert engines["sd35"]["default_steps"] == 20
    assert engines["sd35"]["default_guidance"] == 1.0


def test_get_engine_class_returns_correct_class():
    from imagecli.engines.flux2_klein import Flux2KleinEngine
    from imagecli.engines.flux1_dev import Flux1DevEngine
    from imagecli.engines.sd35 import SD35Engine

    assert get_engine_class("flux2-klein") is Flux2KleinEngine
    assert get_engine_class("flux1-dev") is Flux1DevEngine
    assert get_engine_class("sd35") is SD35Engine


def test_get_engine_class_unknown_raises():
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine_class("nonexistent")


def test_get_engine_returns_instance():
    engine = get_engine("flux2-klein")
    assert engine.name == "flux2-klein"
    assert engine._pipe is None  # not loaded yet


def test_get_engine_unknown_raises():
    with pytest.raises(ValueError, match="Unknown engine"):
        get_engine("unknown-engine")
