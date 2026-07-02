"""Pivotal tuning application to a Klein pipeline.

Wires a parsed ``PivotalEmbedding`` into the tokenizer + text encoder and
monkey-patches ``pipe.encode_prompt`` to expand bare triggers before
tokenization.
"""

from __future__ import annotations

import logging

from imagecli.pivotal_load import PivotalEmbedding

logger = logging.getLogger(__name__)


def _placeholder_tokens_for(pivotal: PivotalEmbedding) -> list[str]:
    """Placeholder token sequence a pivotal expands to (bare trigger + suffixes)."""
    n = pivotal.num_tokens
    return [pivotal.trigger] + [f"{pivotal.trigger}_{i}" for i in range(1, n)]


def apply_pivotals_to_pipe(pipe, pivotals: list[PivotalEmbedding]) -> list[list[int]]:
    """Wire one or more pivotal embeddings into a Flux2Klein pipeline atomically.

    Adds placeholder tokens for every pivotal in a single ``add_tokens`` call,
    resizes the TE embedding table once, writes the trained vectors for each
    pivotal into its rows, and runs a per-pivotal round-trip assertion.
    Returns a list of per-pivotal placeholder id lists (same order as inputs).

    Atomicity: this is a collision *pre-check* guarantee. If any trigger
    (or its ``_1..._{n-1}`` suffixes) collides with an existing vocab entry
    OR with another pivotal's placeholder set, the function raises BEFORE
    calling ``add_tokens`` — every pivotal is applied or none are, with
    no tokenizer mutation. The guarantee does NOT extend past ``add_tokens``:
    if ``resize_token_embeddings`` or the vector-write sequence below fails
    (OOM, backend bug) after ``add_tokens`` succeeded, the tokenizer is
    left dirty and the engine must be discarded via ``cleanup()``.

    Must be called while the text encoder is still on CPU (before any
    ``.to("cuda")``). The tokenizer and TE writes are independent of LoRA
    fuse/unload and of transformer quantization — only the TE embedding
    table is mutated.

    Raises:
      ValueError: ``pivotals`` empty; OR two pivotals share placeholder
        tokens; OR any placeholder collides with the existing vocab.
      RuntimeError: if a round-trip read-back does not match the source
        vectors within ``atol=5e-2`` (bf16 precision bound). Intentionally
        ``raise`` rather than ``assert`` so the guard survives ``python -O``.
    """
    import torch

    if not pivotals:
        raise ValueError("apply_pivotals_to_pipe: pivotals list is empty.")

    tok = pipe.tokenizer
    te = pipe.text_encoder

    per_pivotal_tokens: list[list[str]] = [_placeholder_tokens_for(p) for p in pivotals]
    flat_tokens: list[str] = [t for group in per_pivotal_tokens for t in group]

    # Atomic pre-check (1): two pivotals must not claim the same placeholder.
    seen: set[str] = set()
    intra_dupes: list[str] = []
    for t in flat_tokens:
        if t in seen:
            intra_dupes.append(t)
        seen.add(t)
    if intra_dupes:
        triggers = [p.trigger for p in pivotals]
        raise ValueError(
            f"apply_pivotals_to_pipe: triggers {triggers} share placeholder "
            f"token(s) {sorted(set(intra_dupes))}. Pick distinct trigger words."
        )

    # Atomic pre-check (2): no placeholder may already exist in the vocab.
    # HuggingFace ``add_tokens`` is not atomic on partial collision, so this
    # check must happen before calling it. Checking against both
    # ``added_tokens_encoder`` and the base ``get_vocab()`` keeps the op
    # atomic: either all tokens are fresh and we proceed, or we raise
    # without touching the tokenizer.
    added_encoder = getattr(tok, "added_tokens_encoder", {}) or {}
    base_vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    collisions = [t for t in flat_tokens if t in added_encoder or t in base_vocab]
    if collisions:
        raise ValueError(
            f"apply_pivotals_to_pipe: placeholder token(s) already exist in the "
            f"tokenizer vocabulary: {sorted(set(collisions))}. "
            f"Use different trigger word(s)."
        )

    added = tok.add_tokens(flat_tokens)
    if added != len(flat_tokens):
        # Defensive: unreachable given the pre-check, but surface any tokenizer
        # de-duplication quirk rather than silently corrupting TE rows.
        raise ValueError(
            f"apply_pivotals_to_pipe: pre-check passed but add_tokens returned "
            f"{added}/{len(flat_tokens)}. Tokenizer state may be inconsistent."
        )

    te.resize_token_embeddings(len(tok))
    embed_tokens = te.get_input_embeddings()
    weight = embed_tokens.weight

    out_ids: list[list[int]] = []
    with torch.no_grad():
        for pivotal, group in zip(pivotals, per_pivotal_tokens, strict=True):
            placeholder_ids = [tok.convert_tokens_to_ids(t) for t in group]
            vecs = pivotal.vectors.to(device=weight.device, dtype=weight.dtype)
            for i, pid in enumerate(placeholder_ids):
                weight[pid] = vecs[i]
            out_ids.append(placeholder_ids)

    # Per-pivotal round-trip check. atol=5e-2 accounts for fp32→bf16→fp32
    # precision loss. Intentionally NOT ``assert`` — ``assert`` is stripped
    # under ``python -O`` / ``PYTHONOPTIMIZE=1`` which would defeat the
    # silent-drop prevention design goal.
    for pivotal, placeholder_ids in zip(pivotals, out_ids, strict=True):
        trigger_id = tok.convert_tokens_to_ids(pivotal.trigger)
        if trigger_id != placeholder_ids[0]:
            raise RuntimeError(
                f"pivotal round-trip: trigger id mismatch for {pivotal.trigger!r} "
                f"({trigger_id} vs {placeholder_ids[0]})"
            )
        te_rows = embed_tokens.weight[placeholder_ids].detach().float().cpu()
        src = pivotal.vectors.detach().float().cpu()
        if not torch.allclose(te_rows, src, atol=5e-2):
            raise RuntimeError(
                f"pivotal round-trip: vector mismatch for {pivotal.trigger!r}, "
                f"max abs diff {(te_rows - src).abs().max().item():.3e}"
            )
        logger.info(
            "Pivotal: loaded %d tokens for %r from %s",
            pivotal.num_tokens,
            pivotal.trigger,
            pivotal.source_path,
        )

    return out_ids


def apply_pivotal_to_pipe(pipe, pivotal: PivotalEmbedding) -> list[int]:
    """Singular wrapper — delegates to :func:`apply_pivotals_to_pipe`.

    Kept as the ergonomic entry point for N=1. Behavior is identical to the
    pre-#34 implementation because the plural path handles N=1 through the
    same atomic-mutation code.
    """
    return apply_pivotals_to_pipe(pipe, [pivotal])[0]


def _maybe_convert_prompt(prompt: str, tokenizer) -> str:
    """Expand bare triggers into their multi-vector placeholder form.

    Tokenizer-agnostic — uses only ``tokenizer.tokenize`` and
    ``tokenizer.added_tokens_encoder`` (standard ``PreTrainedTokenizerBase``
    API). Adds a double-expansion warning: if the user manually pre-expanded
    the prompt (``{trigger}_1`` already present), we skip expansion and warn
    rather than double-expanding silently.

    Substring-collision safe: rebuilds the output from the tokenized stream
    rather than using ``str.replace`` on the raw prompt. A word like
    ``"lyrafaces"`` that contains the trigger as a prefix is left untouched,
    because replacement happens at the token level, not the substring level.
    """
    tokens = tokenizer.tokenize(prompt)
    unique = set(tokens)

    # Pre-compute expansions for each triggered token. Map trigger → expansion
    # list (or None if not a trigger / skipped).
    expansions: dict[str, list[str] | None] = {}
    warned_doubles: set[str] = set()
    for token in unique:
        if token not in tokenizer.added_tokens_encoder:
            continue
        suffixes: list[str] = []
        i = 1
        while f"{token}_{i}" in tokenizer.added_tokens_encoder:
            suffixes.append(f"{token}_{i}")
            i += 1
        if not suffixes:
            # Single-vector placeholder — no expansion needed.
            continue
        # Double-expansion guard: if any suffix is already in the tokenized
        # prompt, the user pre-expanded manually. Skip + warn instead of
        # double-expanding (which produces garbage cross-attention).
        if any(s in unique for s in suffixes):
            if token not in warned_doubles:
                logger.warning(
                    "Prompt already contains placeholder %r — possible "
                    "double-expansion. Write the bare trigger once and let "
                    "pivotal.py expand it.",
                    f"{token}_1",
                )
                warned_doubles.add(token)
            continue
        expansions[token] = [token, *suffixes]

    if not expansions:
        return prompt

    # Rebuild the prompt from the tokenized stream, substituting each trigger
    # token with its expansion. Because the tokenizer's split is authoritative,
    # this cannot corrupt substrings inside longer words.
    out_tokens: list[str] = []
    for t in tokens:
        expansion = expansions.get(t)
        if expansion is None:
            out_tokens.append(t)
        else:
            out_tokens.extend(expansion)
    return " ".join(out_tokens)


def _patch_encode_prompt(pipe) -> None:  # pyright: ignore[reportUnusedFunction]
    """Wrap ``pipe.encode_prompt`` to expand bare triggers before tokenization.

    Instance-level monkey-patch (does not touch the class). Idempotent via
    the ``pipe._imagecli_pivotal_patched`` sentinel — calling this twice on
    the same pipe is a no-op, which matters when multiple pivotal LoRAs are
    applied in one load (each call would otherwise stack another wrapper
    around the prior wrapper, visibly corrupting log output and subtly
    altering call semantics).

    Covers all three inference paths that pass a string prompt through
    ``encode_prompt``:
      1. ``ImageEngine.generate`` → ``self._pipe(prompt=...)`` → internal encode
      2. all-on-GPU batch ``encode_and_generate`` → same
      3. 2-phase batch phase-1 ``engine.encode_prompt`` → direct

    The 4th path (``generate_from_embeddings``) passes precomputed
    ``prompt_embeds=`` and bypasses ``encode_prompt`` entirely, but it
    consumes embeddings produced by phase-1, which already went through this
    patch.
    """
    # `is True` (not truthy) so that test doubles like MagicMock, which
    # auto-create attributes returning a truthy child MagicMock on any
    # attribute access, do not spuriously trigger the idempotency guard.
    if getattr(pipe, "_imagecli_pivotal_patched", False) is True:
        return
    original = pipe.encode_prompt
    tokenizer = pipe.tokenizer
    # Closure-carried flag: first rewrite per load logs at INFO, subsequent at
    # DEBUG. Proves the patch fired for the user without spamming batch logs.
    logged_first = [False]

    def _patched(*args, **kwargs):
        # Resolve prompt from kwargs OR args[0], tracking which source so we can
        # re-inject via the same channel (re-injecting a positional as kwarg
        # would silently change the underlying encode_prompt's call signature
        # if its first positional parameter is not named "prompt").
        prompt_is_positional = False
        prompt = kwargs.get("prompt")
        if prompt is None and args:
            prompt = args[0]
            prompt_is_positional = True

        new_value: object = prompt  # default: unchanged
        if isinstance(prompt, str):
            new_prompt = _maybe_convert_prompt(prompt, tokenizer)
            if new_prompt != prompt:
                if not logged_first[0]:
                    logger.info("Pivotal: expanded %r → %r", prompt, new_prompt)
                    logged_first[0] = True
                else:
                    logger.debug("Pivotal: expanded %r → %r", prompt, new_prompt)
            new_value = new_prompt
        elif isinstance(prompt, list):
            new_list = [_maybe_convert_prompt(p, tokenizer) for p in prompt]
            if new_list != prompt and not logged_first[0]:
                logger.info("Pivotal: expanded prompts (list of %d)", len(prompt))
                logged_first[0] = True
            new_value = new_list

        if prompt_is_positional:
            args = (new_value,) + args[1:]
        else:
            kwargs["prompt"] = new_value
        return original(*args, **kwargs)

    pipe.encode_prompt = _patched
    pipe._imagecli_pivotal_patched = True
