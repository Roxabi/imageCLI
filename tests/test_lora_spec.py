"""Tests for imagecli.lora_spec — LoraSpec frozen dataclass."""

from __future__ import annotations

import pytest

from imagecli.lora_spec import LoraSpec


# ── Defaults ─────────────────────────────────────────────────────────────────


def test_lora_spec_defaults():
    # Arrange / Act
    spec = LoraSpec("/x")

    # Assert
    assert spec.path == "/x"
    assert spec.scale == 1.0
    assert spec.trigger is None
    assert spec.embedding_path is None


def test_lora_spec_all_fields():
    # Arrange / Act
    spec = LoraSpec(
        "/path/lora.safetensors", scale=0.7, trigger="lyraface", embedding_path="/emb.st"
    )

    # Assert
    assert spec.path == "/path/lora.safetensors"
    assert spec.scale == 0.7
    assert spec.trigger == "lyraface"
    assert spec.embedding_path == "/emb.st"


# ── Frozen ───────────────────────────────────────────────────────────────────


def test_lora_spec_is_frozen():
    # Arrange
    spec = LoraSpec("/x")

    # Act / Assert — frozen dataclass raises FrozenInstanceError on any assignment
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError (subclass of AttributeError)
        spec.path = "/y"  # type: ignore[misc]


def test_lora_spec_frozen_scale():
    # Arrange
    spec = LoraSpec("/x", scale=0.5)

    # Act / Assert
    with pytest.raises(Exception):
        spec.scale = 1.0  # type: ignore[misc]


# ── Equality ─────────────────────────────────────────────────────────────────


def test_lora_spec_equality_same_fields():
    # Arrange
    spec_a = LoraSpec("/a", scale=1.0, trigger=None, embedding_path=None)
    spec_b = LoraSpec("/a", scale=1.0, trigger=None, embedding_path=None)

    # Assert
    assert spec_a == spec_b


def test_lora_spec_inequality_different_path():
    # Arrange
    spec_a = LoraSpec("/a")
    spec_b = LoraSpec("/b")

    # Assert
    assert spec_a != spec_b


def test_lora_spec_inequality_different_scale():
    # Arrange
    spec_a = LoraSpec("/a", scale=1.0)
    spec_b = LoraSpec("/a", scale=0.5)

    # Assert
    assert spec_a != spec_b


def test_lora_spec_equality_with_trigger_and_embedding():
    # Arrange
    spec_a = LoraSpec("/a", scale=0.8, trigger="t", embedding_path="/e")
    spec_b = LoraSpec("/a", scale=0.8, trigger="t", embedding_path="/e")

    # Assert
    assert spec_a == spec_b


def test_lora_spec_hashable():
    # Arrange — frozen dataclasses are hashable, usable in sets/dict keys
    spec_a = LoraSpec("/a")
    spec_b = LoraSpec("/a")

    # Assert
    assert hash(spec_a) == hash(spec_b)
    assert len({spec_a, spec_b}) == 1
