"""Tests for imagecli.pivotal — loader, validation, apply, prompt expansion."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from imagecli.pivotal import (
    PivotalEmbedding,
    _maybe_convert_prompt,
    _patch_encode_prompt,
    apply_pivotal_to_pipe,
    detect_pivotal_in_lora,
    load_pivotal_embedding,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _write_merged_lora(path: Path, num_tokens: int = 4, hidden: int = 2560) -> Path:
    """Write a fake merged LoRA safetensors: emb_params + some LoRA-ish keys."""
    save_file(
        {
            "emb_params": torch.randn(num_tokens, hidden, dtype=torch.float32),
            "transformer.blocks.0.attn.to_q.lora_A.weight": torch.zeros(
                16, 3072, dtype=torch.float32
            ),
            "transformer.blocks.0.attn.to_q.lora_B.weight": torch.zeros(
                3072, 16, dtype=torch.float32
            ),
        },
        str(path),
    )
    return path


def _write_standalone(
    path: Path,
    trigger: str = "lyraface",
    num_tokens: int = 4,
    hidden: int = 2560,
) -> Path:
    """Write a fake ai-toolkit A1111-format standalone embedding file."""
    metadata = {
        "name": json.dumps(trigger),
        "step": json.dumps(2000),
        "string_to_param": json.dumps({"*": "emb_params"}),
    }
    save_file(
        {"emb_params": torch.randn(num_tokens, hidden, dtype=torch.float32)},
        str(path),
        metadata=metadata,
    )
    return path


def _make_mock_pipe(initial_vocab: int = 151669, hidden: int = 2560):
    """Build a MagicMock pipe with tokenizer + text_encoder behaving like Klein."""
    pipe = MagicMock()

    # Tokenizer — start with empty added_tokens_encoder, grow on add_tokens
    tok_state = {"vocab_size": initial_vocab, "added": {}}

    def _add_tokens(tokens):
        added = 0
        for t in tokens:
            if t not in tok_state["added"]:
                tok_state["added"][t] = tok_state["vocab_size"] + added
                added += 1
        tok_state["vocab_size"] += added
        return added

    def _convert_to_id(t):
        return tok_state["added"].get(t, -1)

    tokenizer = MagicMock()
    tokenizer.add_tokens = MagicMock(side_effect=_add_tokens)
    tokenizer.convert_tokens_to_ids = MagicMock(side_effect=_convert_to_id)
    tokenizer.__len__ = MagicMock(side_effect=lambda: tok_state["vocab_size"])
    tokenizer.tokenize = MagicMock(side_effect=lambda s: s.split())
    tokenizer.added_tokens_encoder = tok_state["added"]
    pipe.tokenizer = tokenizer

    # Text encoder — embed_tokens is a real nn.Embedding so slicing works
    weight_size = initial_vocab + 100  # headroom for added tokens
    embed = torch.nn.Embedding(weight_size, hidden, dtype=torch.bfloat16)
    torch.nn.init.zeros_(embed.weight)
    text_encoder = MagicMock()
    text_encoder.get_input_embeddings = MagicMock(return_value=embed)
    text_encoder.resize_token_embeddings = MagicMock()  # no-op; we pre-sized
    text_encoder.config = MagicMock(hidden_size=hidden)
    pipe.text_encoder = text_encoder

    # Real encode_prompt stub that records calls — for _patch_encode_prompt tests
    pipe.encode_prompt = MagicMock(return_value=("dummy_embeds", "dummy_ids"))

    return pipe


# ── Loader tests (V1 RED: T1.6) ───────────────────────────────────────────


def test_load_merged_format(tmp_path: Path):
    path = _write_merged_lora(tmp_path / "lora.safetensors", num_tokens=4)
    piv = load_pivotal_embedding(path, trigger="lyraface")
    assert piv is not None
    assert piv.trigger == "lyraface"
    assert piv.num_tokens == 4
    assert piv.vectors.shape == (4, 2560)
    assert piv.source == "merged"
    assert piv.source_path == path


def test_load_standalone_format_with_explicit_trigger(tmp_path: Path):
    emb_path = _write_standalone(tmp_path / "lyraface2000.safetensors", num_tokens=4)
    piv = load_pivotal_embedding(
        lora_path=None, trigger="lyraface", embedding_path=emb_path
    )
    assert piv is not None
    assert piv.trigger == "lyraface"
    assert piv.source == "standalone"
    assert piv.num_tokens == 4
    assert piv.source_path == emb_path


def test_load_standalone_format_trigger_inferred_from_metadata(tmp_path: Path):
    emb_path = _write_standalone(tmp_path / "inferred2000.safetensors", trigger="infrd")
    piv = load_pivotal_embedding(
        lora_path=None, trigger=None, embedding_path=emb_path
    )
    assert piv is not None
    assert piv.trigger == "infrd"


def test_load_returns_none_when_no_pivotal(tmp_path: Path):
    # Plain LoRA without emb_params
    path = tmp_path / "plain_lora.safetensors"
    save_file(
        {"transformer.blocks.0.lora_A.weight": torch.zeros(16, 3072)},
        str(path),
    )
    piv = load_pivotal_embedding(path, trigger=None)
    assert piv is None


def test_detect_pivotal_in_lora_positive(tmp_path: Path):
    path = _write_merged_lora(tmp_path / "lora.safetensors")
    assert detect_pivotal_in_lora(path) is True


def test_detect_pivotal_in_lora_negative(tmp_path: Path):
    path = tmp_path / "plain.safetensors"
    save_file({"some.weight": torch.zeros(4, 4)}, str(path))
    assert detect_pivotal_in_lora(path) is False


# ── Shape validation tests (V1 RED: T1.7) ─────────────────────────────────


def _write_shaped_lora(path: Path, shape: tuple, dtype=torch.float32) -> Path:
    save_file({"emb_params": torch.zeros(shape, dtype=dtype)}, str(path))
    return path


def test_validate_rejects_ndim_not_2(tmp_path: Path):
    # 1D tensor
    save_file(
        {"emb_params": torch.zeros(2560, dtype=torch.float32)},
        str(tmp_path / "bad.safetensors"),
    )
    with pytest.raises(ValueError, match="ndim"):
        load_pivotal_embedding(tmp_path / "bad.safetensors", trigger="x")


def test_validate_rejects_zero_n(tmp_path: Path):
    # Empty along first axis — use shape (0, 2560)
    tensor = torch.zeros(0, 2560, dtype=torch.float32)
    save_file({"emb_params": tensor}, str(tmp_path / "empty.safetensors"))
    with pytest.raises(ValueError, match=r"N >= 1"):
        load_pivotal_embedding(tmp_path / "empty.safetensors", trigger="x")


def test_validate_rejects_n_over_cap(tmp_path: Path):
    path = _write_shaped_lora(tmp_path / "huge.safetensors", (64, 2560))
    with pytest.raises(ValueError, match="N <= 32"):
        load_pivotal_embedding(path, trigger="x")


def test_validate_rejects_wrong_hidden(tmp_path: Path):
    path = _write_shaped_lora(tmp_path / "wrong.safetensors", (4, 3072))
    with pytest.raises(ValueError, match="hidden_size"):
        load_pivotal_embedding(path, trigger="x", te_hidden_size=2560)


# ── Missing-trigger hard error (V1 RED: T1.9) ─────────────────────────────


def test_missing_trigger_hard_error(tmp_path: Path):
    path = _write_merged_lora(tmp_path / "lora.safetensors")
    with pytest.raises(ValueError) as exc_info:
        load_pivotal_embedding(path, trigger=None)
    msg = str(exc_info.value)
    assert "--trigger" in msg
    assert "silently ignored" in msg or "silent" in msg.lower()


# ── _maybe_convert_prompt tests (V1 RED: T1.10) ───────────────────────────


def _make_tok_with_added(tokens: list[str]):
    tok = MagicMock()
    tok.added_tokens_encoder = {t: 151669 + i for i, t in enumerate(tokens)}
    tok.tokenize = lambda s: s.split()
    return tok


def test_maybe_convert_prompt_multi_vector():
    tok = _make_tok_with_added(
        ["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"]
    )
    out = _maybe_convert_prompt("lyraface in space", tok)
    assert out == "lyraface lyraface_1 lyraface_2 lyraface_3 in space"


def test_maybe_convert_prompt_no_trigger_present():
    tok = _make_tok_with_added(
        ["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"]
    )
    out = _maybe_convert_prompt("just a plain prompt", tok)
    assert out == "just a plain prompt"


def test_maybe_convert_prompt_empty_added_tokens():
    tok = _make_tok_with_added([])
    out = _maybe_convert_prompt("any prompt", tok)
    assert out == "any prompt"


def test_maybe_convert_prompt_single_vector_no_expansion():
    # Trigger with no _1 suffix → single-vector, no expansion
    tok = _make_tok_with_added(["singletok"])
    out = _maybe_convert_prompt("singletok cat", tok)
    assert out == "singletok cat"


def test_maybe_convert_prompt_double_expansion_warns(caplog):
    tok = _make_tok_with_added(
        ["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"]
    )
    with caplog.at_level(logging.WARNING):
        out = _maybe_convert_prompt("lyraface lyraface_1 cat", tok)
    # Should NOT double-expand
    assert out == "lyraface lyraface_1 cat"
    assert "double-expansion" in caplog.text.lower()


def test_maybe_convert_prompt_substring_collision():
    """Catches the str.replace substring hazard: trigger 'lyraface' must NOT
    corrupt the longer word 'lyrafaces' when the tokenizer returns 'lyrafaces'
    as a distinct token (i.e. 'lyraface' is not in the tokenized output).
    """
    tok = _make_tok_with_added(
        ["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"]
    )
    # Tokenizer mock splits on whitespace, so "lyrafaces" is one token and is
    # NOT in added_tokens_encoder → expansion should not fire.
    out = _maybe_convert_prompt("lyrafaces in space", tok)
    assert out == "lyrafaces in space", (
        f"substring collision: 'lyrafaces' was corrupted by str.replace expansion. "
        f"Got: {out!r}"
    )


def test_maybe_convert_prompt_substring_collision_when_trigger_tokenized():
    """Harder case: if 'lyraface' IS present as a token in the same prompt
    alongside a substring-containing word, the expansion must only replace
    whole tokens, not substrings inside other words.
    """
    tok = _make_tok_with_added(
        ["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"]
    )
    # "lyraface" is a real trigger word here, "lyrafaces" is a different token
    # that happens to contain "lyraface" as a prefix. The trigger should expand,
    # but "lyrafaces" must stay intact.
    out = _maybe_convert_prompt("lyraface and lyrafaces coexist", tok)
    # The trigger expands; "lyrafaces" is untouched.
    assert out == (
        "lyraface lyraface_1 lyraface_2 lyraface_3 and lyrafaces coexist"
    ), f"substring collision during expansion: got {out!r}"


# ── apply_pivotal_to_pipe tests (V2 RED: T2.9) ────────────────────────────


def test_apply_adds_exactly_n_tokens(tmp_path: Path):
    pipe = _make_mock_pipe()
    piv = PivotalEmbedding(
        trigger="lyraface",
        vectors=torch.randn(4, 2560, dtype=torch.float32),
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.safetensors",
    )
    ids = apply_pivotal_to_pipe(pipe, piv)
    assert len(ids) == 4
    # Must have been called once with the canonical name list
    pipe.tokenizer.add_tokens.assert_called_once()
    call_arg = pipe.tokenizer.add_tokens.call_args[0][0]
    assert call_arg == ["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"]


def test_apply_round_trip_assertion_passes(tmp_path: Path):
    pipe = _make_mock_pipe()
    # Fixed seed: torch.randn tails (~4σ) can occasionally produce values where
    # bf16 round-trip exceeds 5e-2 across 10240 elements. Deterministic seed
    # eliminates that flake risk without weakening the test.
    torch.manual_seed(42)
    vecs = torch.randn(4, 2560, dtype=torch.float32)
    piv = PivotalEmbedding(
        trigger="lyraface",
        vectors=vecs,
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.safetensors",
    )
    ids = apply_pivotal_to_pipe(pipe, piv)  # Should not raise
    # Verify vectors actually landed in the embedding table at the right rows
    embed_weight = pipe.text_encoder.get_input_embeddings().weight
    for i, pid in enumerate(ids):
        # bf16 round-trip: compare with loose tolerance (matches prod atol=5e-2)
        assert torch.allclose(
            embed_weight[pid].float().cpu(), vecs[i].float().cpu(), atol=5e-2
        )


def test_apply_rejects_existing_trigger(tmp_path: Path):
    pipe = _make_mock_pipe()
    # Pre-populate the added_tokens_encoder with the trigger
    pipe.tokenizer.added_tokens_encoder["lyraface"] = 99999
    # add_tokens should now return fewer than N
    pipe.tokenizer.add_tokens.side_effect = lambda toks: len(toks) - 1

    piv = PivotalEmbedding(
        trigger="lyraface",
        vectors=torch.randn(4, 2560, dtype=torch.float32),
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.safetensors",
    )
    with pytest.raises(ValueError, match="already exist"):
        apply_pivotal_to_pipe(pipe, piv)


def test_apply_independent_of_unload_lora_weights(tmp_path: Path):
    """SC-5b: unload_lora_weights is a no-op for tokenizer + TE state."""
    pipe = _make_mock_pipe()
    piv = PivotalEmbedding(
        trigger="lyraface",
        vectors=torch.randn(4, 2560, dtype=torch.float32),
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.safetensors",
    )
    ids = apply_pivotal_to_pipe(pipe, piv)

    # Simulate unload_lora_weights being called (it would normally only touch
    # transformer PEFT adapters, not tokenizer/TE)
    pipe.unload_lora_weights = MagicMock()  # no-op
    pipe.unload_lora_weights()

    # Verify tokenizer and TE state survived
    assert pipe.tokenizer.added_tokens_encoder["lyraface"] == ids[0]
    embed_weight = pipe.text_encoder.get_input_embeddings().weight
    # First trained vector still present
    assert embed_weight[ids[0]].abs().sum().item() > 0


# ── _patch_encode_prompt tests (V2 RED: T2.5) ─────────────────────────────


def test_patch_encode_prompt_expands_string_prompt(tmp_path: Path):
    pipe = _make_mock_pipe()
    # First wire up pivotal so tokenizer knows the placeholders
    piv = PivotalEmbedding(
        trigger="lyraface",
        vectors=torch.randn(4, 2560, dtype=torch.float32),
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.safetensors",
    )
    apply_pivotal_to_pipe(pipe, piv)

    # Replace encode_prompt with a recorder before patching
    recorded = {}

    def _record(*args, **kwargs):
        recorded["prompt"] = kwargs.get("prompt") or (args[0] if args else None)
        return ("embeds", "ids")

    pipe.encode_prompt = _record
    _patch_encode_prompt(pipe)

    pipe.encode_prompt(prompt="lyraface sitting on a bench")
    assert (
        recorded["prompt"]
        == "lyraface lyraface_1 lyraface_2 lyraface_3 sitting on a bench"
    )


def test_patch_encode_prompt_handles_list_input(tmp_path: Path):
    pipe = _make_mock_pipe()
    piv = PivotalEmbedding(
        trigger="lyraface",
        vectors=torch.randn(4, 2560, dtype=torch.float32),
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.safetensors",
    )
    apply_pivotal_to_pipe(pipe, piv)

    recorded = {}

    def _record(*args, **kwargs):
        recorded["prompt"] = kwargs.get("prompt")
        return ("embeds", "ids")

    pipe.encode_prompt = _record
    _patch_encode_prompt(pipe)

    pipe.encode_prompt(prompt=["lyraface cat", "lyraface dog"])
    assert recorded["prompt"] == [
        "lyraface lyraface_1 lyraface_2 lyraface_3 cat",
        "lyraface lyraface_1 lyraface_2 lyraface_3 dog",
    ]


def test_patch_encode_prompt_passthrough_when_no_trigger(tmp_path: Path):
    pipe = _make_mock_pipe()
    piv = PivotalEmbedding(
        trigger="lyraface",
        vectors=torch.randn(4, 2560, dtype=torch.float32),
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.safetensors",
    )
    apply_pivotal_to_pipe(pipe, piv)

    recorded = {}

    def _record(*args, **kwargs):
        recorded["prompt"] = kwargs.get("prompt")
        return ("embeds", "ids")

    pipe.encode_prompt = _record
    _patch_encode_prompt(pipe)

    pipe.encode_prompt(prompt="a plain cat on a bench")
    # No trigger in prompt → passthrough unchanged
    assert recorded["prompt"] == "a plain cat on a bench"


def test_patch_encode_prompt_handles_positional_arg(tmp_path: Path):
    """Ensure the _patched closure correctly expands a prompt passed as a
    positional argument (not keyword), and re-injects it at the same position
    rather than converting it to a kwarg — which would silently change the
    call signature seen by the underlying encode_prompt.
    """
    pipe = _make_mock_pipe()
    piv = PivotalEmbedding(
        trigger="lyraface",
        vectors=torch.randn(4, 2560, dtype=torch.float32),
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.safetensors",
    )
    apply_pivotal_to_pipe(pipe, piv)

    recorded = {}

    def _record(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return ("embeds", "ids")

    pipe.encode_prompt = _record
    _patch_encode_prompt(pipe)

    # Positional call — prompt is args[0], not a kwarg
    pipe.encode_prompt("lyraface cat")

    # The expanded prompt must arrive via args (positional), not kwargs
    assert recorded["args"] == (
        "lyraface lyraface_1 lyraface_2 lyraface_3 cat",
    ), f"positional prompt was re-routed to kwargs: args={recorded['args']}, kwargs={recorded['kwargs']}"
    assert "prompt" not in recorded["kwargs"], (
        "positional prompt must not leak into kwargs"
    )


# ── Missing-file error path tests ─────────────────────────────────────────


def test_detect_pivotal_in_lora_missing_file(tmp_path: Path):
    """`detect_pivotal_in_lora` should propagate FileNotFoundError (or the
    safetensors-equivalent) when called on a non-existent path.
    """
    ghost = tmp_path / "ghost.safetensors"
    with pytest.raises((FileNotFoundError, OSError)):
        detect_pivotal_in_lora(ghost)


def test_load_pivotal_embedding_standalone_missing_file(tmp_path: Path):
    """Explicit `embedding_path` pointing at a non-existent file should raise."""
    ghost = tmp_path / "ghost.safetensors"
    with pytest.raises((FileNotFoundError, OSError)):
        load_pivotal_embedding(
            lora_path=None,
            trigger="lyraface",
            embedding_path=ghost,
        )
