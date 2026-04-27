"""Tests for imagecli.pivotal — loader, validation, apply, prompt expansion."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from imagecli.pivotal import (
    PivotalEmbedding,
    _maybe_convert_prompt,
    _patch_encode_prompt,
    apply_pivotal_to_pipe,
    apply_pivotals_to_pipe,
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
    tok_added: dict[str, int] = {}
    tok_state = {"vocab_size": initial_vocab}

    def _add_tokens(tokens):
        added = 0
        for t in tokens:
            if t not in tok_added:
                tok_added[t] = tok_state["vocab_size"] + added
                added += 1
        tok_state["vocab_size"] += added
        return added

    def _convert_to_id(t):
        return tok_added.get(t, -1)

    tokenizer = MagicMock()
    tokenizer.add_tokens = MagicMock(side_effect=_add_tokens)
    tokenizer.convert_tokens_to_ids = MagicMock(side_effect=_convert_to_id)
    tokenizer.__len__ = MagicMock(side_effect=lambda: tok_state["vocab_size"])
    tokenizer.tokenize = MagicMock(side_effect=lambda s: s.split())
    tokenizer.added_tokens_encoder = tok_added
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
    piv = load_pivotal_embedding(lora_path=None, trigger="lyraface", embedding_path=emb_path)
    assert piv is not None
    assert piv.trigger == "lyraface"
    assert piv.source == "standalone"
    assert piv.num_tokens == 4
    assert piv.source_path == emb_path


def test_load_standalone_format_trigger_inferred_from_metadata(tmp_path: Path):
    emb_path = _write_standalone(tmp_path / "inferred2000.safetensors", trigger="infrd")
    piv = load_pivotal_embedding(lora_path=None, trigger=None, embedding_path=emb_path)
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
    tok = _make_tok_with_added(["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"])
    out = _maybe_convert_prompt("lyraface in space", tok)
    assert out == "lyraface lyraface_1 lyraface_2 lyraface_3 in space"


def test_maybe_convert_prompt_no_trigger_present():
    tok = _make_tok_with_added(["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"])
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
    tok = _make_tok_with_added(["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"])
    with caplog.at_level(logging.WARNING, logger="imagecli.pivotal_apply"):
        out = _maybe_convert_prompt("lyraface lyraface_1 cat", tok)
    # Should NOT double-expand
    assert out == "lyraface lyraface_1 cat"
    assert "double-expansion" in caplog.text.lower()
    assert any(r.name == "imagecli.pivotal_apply" for r in caplog.records)


def test_maybe_convert_prompt_substring_collision():
    """Catches the str.replace substring hazard: trigger 'lyraface' must NOT
    corrupt the longer word 'lyrafaces' when the tokenizer returns 'lyrafaces'
    as a distinct token (i.e. 'lyraface' is not in the tokenized output).
    """
    tok = _make_tok_with_added(["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"])
    # Tokenizer mock splits on whitespace, so "lyrafaces" is one token and is
    # NOT in added_tokens_encoder → expansion should not fire.
    out = _maybe_convert_prompt("lyrafaces in space", tok)
    assert out == "lyrafaces in space", (
        f"substring collision: 'lyrafaces' was corrupted by str.replace expansion. Got: {out!r}"
    )


def test_maybe_convert_prompt_substring_collision_when_trigger_tokenized():
    """Harder case: if 'lyraface' IS present as a token in the same prompt
    alongside a substring-containing word, the expansion must only replace
    whole tokens, not substrings inside other words.
    """
    tok = _make_tok_with_added(["lyraface", "lyraface_1", "lyraface_2", "lyraface_3"])
    # "lyraface" is a real trigger word here, "lyrafaces" is a different token
    # that happens to contain "lyraface" as a prefix. The trigger should expand,
    # but "lyrafaces" must stay intact.
    out = _maybe_convert_prompt("lyraface and lyrafaces coexist", tok)
    # The trigger expands; "lyrafaces" is untouched.
    assert out == ("lyraface lyraface_1 lyraface_2 lyraface_3 and lyrafaces coexist"), (
        f"substring collision during expansion: got {out!r}"
    )


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
        assert torch.allclose(embed_weight[pid].float().cpu(), vecs[i].float().cpu(), atol=5e-2)


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
    assert recorded["prompt"] == "lyraface lyraface_1 lyraface_2 lyraface_3 sitting on a bench"


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
    assert recorded["args"] == ("lyraface lyraface_1 lyraface_2 lyraface_3 cat",), (
        f"positional prompt was re-routed to kwargs: args={recorded['args']}, kwargs={recorded['kwargs']}"
    )
    assert "prompt" not in recorded["kwargs"], "positional prompt must not leak into kwargs"


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


# ── Shim re-export surface ────────────────────────────────────────────────


def test_shim_reexports_are_identical():
    """`imagecli.pivotal` is a re-export shim over `pivotal_load` + `pivotal_apply`.
    Assert every re-exported symbol resolves to the same object as its source
    module — catches silent drift (e.g. a symbol moved between sub-modules
    without updating the shim, or the shim deleted entirely).
    """
    from imagecli import pivotal, pivotal_apply, pivotal_load

    assert pivotal.PivotalEmbedding is pivotal_load.PivotalEmbedding
    assert pivotal.detect_pivotal_in_lora is pivotal_load.detect_pivotal_in_lora
    assert pivotal.load_pivotal_embedding is pivotal_load.load_pivotal_embedding
    assert pivotal.apply_pivotal_to_pipe is pivotal_apply.apply_pivotal_to_pipe
    assert pivotal._maybe_convert_prompt is pivotal_apply._maybe_convert_prompt
    assert pivotal._patch_encode_prompt is pivotal_apply._patch_encode_prompt


# ── Integration sentinel: Flux2KleinPipeline class structure ──────────────


def test_flux2_klein_pipeline_class_structure_sentinel():
    """Catch diffusers attribute renames that would silently break the pivotal
    engine hooks. imageCLI's apply_pivotal_to_pipe + _patch_encode_prompt rely
    on specific attribute names on a Flux2KleinPipeline instance:

      - pipe.tokenizer            (Qwen2Tokenizer)
      - pipe.text_encoder         (Qwen3Model)
      - pipe.encode_prompt(...)   (instance method, monkey-patched)

    If a future diffusers upgrade renames any of these (e.g. tokenizer →
    tokenizer_2, text_encoder → t5_encoder, encode_prompt → _encode_text),
    the MagicMock-based unit tests above would still pass but production
    would crash at load time. This sentinel checks the real Flux2KleinPipeline
    class structure without loading any weights, so it runs cheaply in CI and
    fails fast on attribute drift.
    """
    import inspect

    from diffusers import Flux2KleinPipeline

    # Constructor parameters — these are the registered modules that the
    # pipeline expects at instantiation. imageCLI's engine hooks access
    # `self._pipe.tokenizer` and `self._pipe.text_encoder` after from_pretrained
    # returns, so both names must be present here.
    init_params = set(inspect.signature(Flux2KleinPipeline.__init__).parameters)
    assert "tokenizer" in init_params, (
        f"Flux2KleinPipeline.__init__ no longer accepts 'tokenizer' — "
        f"pivotal hooks will break. Got params: {init_params}"
    )
    assert "text_encoder" in init_params, (
        f"Flux2KleinPipeline.__init__ no longer accepts 'text_encoder' — "
        f"pivotal hooks will break. Got params: {init_params}"
    )
    assert "transformer" in init_params, (
        f"Flux2KleinPipeline.__init__ no longer accepts 'transformer' — "
        f"engine load paths will break. Got params: {init_params}"
    )

    # encode_prompt must be a method on the pipeline class — _patch_encode_prompt
    # replaces it at the instance level, so the class must expose it as the
    # original callable.
    assert hasattr(Flux2KleinPipeline, "encode_prompt"), (
        "Flux2KleinPipeline no longer has an 'encode_prompt' method — "
        "_patch_encode_prompt monkey-patch target is missing."
    )
    assert callable(getattr(Flux2KleinPipeline, "encode_prompt", None)), (
        "Flux2KleinPipeline.encode_prompt is not callable."
    )

    # Must inherit LoRA loading (used by the engines before pivotal hooks fire)
    # and be a DiffusionPipeline subclass (MRO sanity).
    from diffusers.loaders.lora_pipeline import Flux2LoraLoaderMixin
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    assert issubclass(Flux2KleinPipeline, DiffusionPipeline), (
        "Flux2KleinPipeline no longer inherits DiffusionPipeline — "
        "pipeline class resolution may have shifted."
    )
    assert issubclass(Flux2KleinPipeline, Flux2LoraLoaderMixin), (
        "Flux2KleinPipeline no longer inherits Flux2LoraLoaderMixin — "
        "the pre-pivotal LoRA fuse path may break."
    )


# ── apply_pivotals_to_pipe — plural atomic entry point (#34) ─────────────────


def _make_pivotal(trigger: str, num_tokens: int = 4, hidden: int = 2560, tmp_path=None):
    """Helper: build a PivotalEmbedding with deterministic random vectors."""
    torch.manual_seed(abs(hash(trigger)) % (2**31))
    return PivotalEmbedding(
        trigger=trigger,
        vectors=torch.randn(num_tokens, hidden, dtype=torch.float32),
        num_tokens=num_tokens,
        source="merged",
        source_path=Path(f"/tmp/{trigger}.safetensors")
        if tmp_path is None
        else tmp_path / f"{trigger}.st",
    )


def test_apply_pivotals_empty_raises():
    # Arrange
    pipe = _make_mock_pipe()

    # Act / Assert — empty list must raise before any tokenizer mutation
    with pytest.raises(ValueError, match="empty"):
        apply_pivotals_to_pipe(pipe, [])


def test_apply_pivotals_empty_does_not_call_add_tokens():
    # Arrange
    pipe = _make_mock_pipe()

    # Act
    with pytest.raises(ValueError):
        apply_pivotals_to_pipe(pipe, [])

    # Assert: tokenizer was never touched
    pipe.tokenizer.add_tokens.assert_not_called()


def test_apply_pivotals_n2_happy_path(tmp_path: Path):
    # Arrange
    pipe = _make_mock_pipe()
    piv_a = _make_pivotal("lyraface", tmp_path=tmp_path)
    piv_b = _make_pivotal("mirrorface", tmp_path=tmp_path)

    # Act
    out_ids = apply_pivotals_to_pipe(pipe, [piv_a, piv_b])

    # Assert: returned structure — 2 per-pivotal id lists
    assert len(out_ids) == 2
    assert len(out_ids[0]) == 4  # lyraface: 4 tokens
    assert len(out_ids[1]) == 4  # mirrorface: 4 tokens

    # add_tokens called exactly once with the flat concatenated list
    pipe.tokenizer.add_tokens.assert_called_once()
    flat_arg = pipe.tokenizer.add_tokens.call_args[0][0]
    assert flat_arg == [
        "lyraface",
        "lyraface_1",
        "lyraface_2",
        "lyraface_3",
        "mirrorface",
        "mirrorface_1",
        "mirrorface_2",
        "mirrorface_3",
    ]

    # resize_token_embeddings called exactly once
    pipe.text_encoder.resize_token_embeddings.assert_called_once()

    # Round-trip: vectors land in the embedding table at returned ids
    embed_weight = pipe.text_encoder.get_input_embeddings().weight
    for piv, ids in zip([piv_a, piv_b], out_ids):
        for i, pid in enumerate(ids):
            assert torch.allclose(
                embed_weight[pid].float().cpu(),
                piv.vectors[i].float().cpu(),
                atol=5e-2,
            )


def test_apply_pivotals_shared_trigger_raises_before_add_tokens(tmp_path: Path):
    # Arrange — two pivotals both using trigger "lyra" → intra-set duplicate
    pipe = _make_mock_pipe()
    piv_a = _make_pivotal("lyra", tmp_path=tmp_path)
    piv_b = _make_pivotal("lyra", tmp_path=tmp_path)

    # Snapshot tokenizer + TE state; atomicity = zero mutation on raise
    pre_added = dict(pipe.tokenizer.added_tokens_encoder)
    pre_len = len(pipe.tokenizer)

    # Act / Assert
    with pytest.raises(ValueError, match="share placeholder"):
        apply_pivotals_to_pipe(pipe, [piv_a, piv_b])

    # Guarantee: add_tokens / resize must NOT have been called
    pipe.tokenizer.add_tokens.assert_not_called()
    pipe.text_encoder.resize_token_embeddings.assert_not_called()
    assert dict(pipe.tokenizer.added_tokens_encoder) == pre_added
    assert len(pipe.tokenizer) == pre_len


def test_apply_pivotals_vocab_collision_added_tokens_raises_before_add_tokens(tmp_path: Path):
    # Arrange — trigger already present in added_tokens_encoder
    pipe = _make_mock_pipe()
    pipe.tokenizer.added_tokens_encoder["lyraface"] = 99999
    piv = _make_pivotal("lyraface", tmp_path=tmp_path)

    pre_added = dict(pipe.tokenizer.added_tokens_encoder)
    pre_len = len(pipe.tokenizer)

    # Act / Assert
    with pytest.raises(ValueError, match="already exist"):
        apply_pivotals_to_pipe(pipe, [piv])

    # Guarantee: add_tokens / resize must NOT have been called
    pipe.tokenizer.add_tokens.assert_not_called()
    pipe.text_encoder.resize_token_embeddings.assert_not_called()
    assert dict(pipe.tokenizer.added_tokens_encoder) == pre_added
    assert len(pipe.tokenizer) == pre_len


def test_apply_pivotals_vocab_collision_base_vocab_raises_before_add_tokens(tmp_path: Path):
    # Arrange — trigger in base get_vocab() (not added_tokens_encoder)
    pipe = _make_mock_pipe()
    # Override get_vocab to return the trigger as a pre-existing token
    pipe.tokenizer.get_vocab = MagicMock(return_value={"lyraface": 12345})
    # Clear added_tokens_encoder so collision comes only from base vocab
    pipe.tokenizer.added_tokens_encoder = {}
    piv = _make_pivotal("lyraface", tmp_path=tmp_path)

    pre_added = dict(pipe.tokenizer.added_tokens_encoder)
    pre_len = len(pipe.tokenizer)

    # Act / Assert
    with pytest.raises(ValueError, match="already exist"):
        apply_pivotals_to_pipe(pipe, [piv])

    pipe.tokenizer.add_tokens.assert_not_called()
    pipe.text_encoder.resize_token_embeddings.assert_not_called()
    assert dict(pipe.tokenizer.added_tokens_encoder) == pre_added
    assert len(pipe.tokenizer) == pre_len


def test_apply_pivotal_singular_wrapper_returns_flat_list(tmp_path: Path):
    # Arrange — singular wrapper must return list[int], not list[list[int]]
    pipe = _make_mock_pipe()
    piv = _make_pivotal("lyraface", tmp_path=tmp_path)

    # Act
    result = apply_pivotal_to_pipe(pipe, piv)

    # Assert: flat list, not nested
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(x, int) for x in result)


def test_apply_pivotal_singular_identical_to_plural_n1(tmp_path: Path):
    # Arrange — singular result must equal plural[0] for N=1
    pipe_s = _make_mock_pipe()
    pipe_p = _make_mock_pipe()
    torch.manual_seed(7)
    vecs = torch.randn(4, 2560, dtype=torch.float32)
    piv_s = PivotalEmbedding(
        trigger="lyraface",
        vectors=vecs,
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.st",
    )
    piv_p = PivotalEmbedding(
        trigger="lyraface",
        vectors=vecs,
        num_tokens=4,
        source="merged",
        source_path=tmp_path / "x.st",
    )

    # Act
    ids_singular = apply_pivotal_to_pipe(pipe_s, piv_s)
    ids_plural = apply_pivotals_to_pipe(pipe_p, [piv_p])[0]

    # Assert: same length (ids are assigned sequentially from initial_vocab so equal)
    assert len(ids_singular) == len(ids_plural)


# ── _patch_encode_prompt — idempotency and MagicMock sentinel safety (#34) ───


def test_patch_encode_prompt_idempotent_wraps_once(tmp_path: Path):
    # Arrange — apply a pivotal so tokenizer has tokens, then patch twice
    pipe = _make_mock_pipe()
    piv = _make_pivotal("lyraface", tmp_path=tmp_path)
    apply_pivotals_to_pipe(pipe, [piv])

    pipe.encode_prompt = MagicMock(return_value=("embeds", "ids"))
    _patch_encode_prompt(pipe)
    first_patched = pipe.encode_prompt  # the wrapper after 1st call

    _patch_encode_prompt(pipe)  # 2nd call — must be a no-op
    second_patched = pipe.encode_prompt  # must be the same object

    # Assert: identity preserved — no double-wrapping
    assert first_patched is second_patched


def test_patch_encode_prompt_magicmock_pipe_is_patched():
    # Arrange — fresh MagicMock pipe (attrs auto-created as truthy child MagicMocks)
    pipe = MagicMock()
    # MagicMock auto-creates pipe._imagecli_pivotal_patched as a child MagicMock
    # (truthy), but the `is True` guard must NOT be fooled by it.

    original_encode = MagicMock(return_value=("embeds", "ids"))
    pipe.encode_prompt = original_encode
    pipe.tokenizer.added_tokens_encoder = {}
    pipe.tokenizer.tokenize = MagicMock(side_effect=lambda s: s.split())

    # Act — should NOT be blocked by the sentinel guard
    _patch_encode_prompt(pipe)

    # Assert: pipe was patched (encode_prompt is now a different callable)
    assert pipe.encode_prompt is not original_encode
    assert pipe._imagecli_pivotal_patched is True


# ── Round-trip failure path — raise not assert (#34) ─────────────────────────


def test_apply_pivotals_round_trip_failure_raises_runtime_error(tmp_path: Path):
    # Arrange — mock torch.allclose to return False to force round-trip failure.
    # torch is imported inside the function body, so patch the torch module directly.
    pipe = _make_mock_pipe()
    piv = _make_pivotal("lyraface", tmp_path=tmp_path)

    import torch as _torch

    with patch.object(_torch, "allclose", return_value=False):
        # Act / Assert — must be RuntimeError, not AssertionError
        with pytest.raises(RuntimeError, match="round-trip"):
            apply_pivotals_to_pipe(pipe, [piv])


def test_apply_pivotals_round_trip_uses_raise_not_assert():
    # Arrange — static check: source must use `raise RuntimeError`, not `assert`
    import ast
    import inspect

    from imagecli import pivotal_apply

    source = inspect.getsource(pivotal_apply.apply_pivotals_to_pipe)
    tree = ast.parse(source)

    # Collect all Raise nodes targeting RuntimeError in the function body
    raise_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and node.exc is not None
        and isinstance(node.exc, ast.Call)
        and (
            (isinstance(node.exc.func, ast.Name) and node.exc.func.id == "RuntimeError")
            or (isinstance(node.exc.func, ast.Attribute) and node.exc.func.attr == "RuntimeError")
        )
    ]
    assert raise_nodes, (
        "apply_pivotals_to_pipe must use `raise RuntimeError(...)` for round-trip "
        "failure — `assert` is stripped under `python -O` and would silently pass."
    )
