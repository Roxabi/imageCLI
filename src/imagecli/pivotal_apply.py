"""Pivotal tuning application to a Klein pipeline.

Wires a parsed ``PivotalEmbedding`` into the tokenizer + text encoder and
monkey-patches ``pipe.encode_prompt`` to expand bare triggers before
tokenization.
"""

from __future__ import annotations

import logging

from imagecli.pivotal_load import PivotalEmbedding

logger = logging.getLogger(__name__)


def apply_pivotal_to_pipe(pipe, pivotal: PivotalEmbedding) -> list[int]:
    """Wire a parsed pivotal embedding into a Flux2Klein pipeline.

    Adds placeholder tokens to the tokenizer, resizes the TE's input
    embedding, writes the trained vectors into the new rows, and runs a
    deterministic round-trip assertion. Returns the list of placeholder
    token ids.

    Must be called while the text encoder is still on CPU (before any
    ``.to("cuda")``). The tokenizer and TE writes are independent of LoRA
    fuse/unload and of transformer quantization — only the TE embedding
    table is mutated.

    Raises:
      ValueError: if the trigger or its suffixes already exist in the
        tokenizer vocabulary.
      RuntimeError: if the round-trip read-back does not match the source
        vectors within ``atol=5e-2`` (bf16 precision bound). Intentionally
        ``raise`` rather than ``assert`` so the guard survives ``python -O``.
    """
    import torch

    tok = pipe.tokenizer
    te = pipe.text_encoder
    trigger = pivotal.trigger
    n = pivotal.num_tokens
    placeholder_tokens = [trigger] + [f"{trigger}_{i}" for i in range(1, n)]

    # Pre-check for collisions BEFORE calling add_tokens. HuggingFace add_tokens
    # is not atomic — if some tokens already exist and others don't, the new
    # ones are added anyway and we only learn about the conflict via the return
    # count. That leaves the tokenizer partially mutated, which is confusing to
    # diagnose. Checking against added_tokens_encoder and the base vocab up-front
    # keeps the operation atomic: either all tokens are fresh and we proceed, or
    # we raise without touching the tokenizer.
    added_encoder = getattr(tok, "added_tokens_encoder", {}) or {}
    base_vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    collisions = [
        t for t in placeholder_tokens if t in added_encoder or t in base_vocab
    ]
    if collisions:
        raise ValueError(
            f"Trigger {trigger!r} (or its suffixes) already exist in the "
            f"tokenizer vocabulary: {collisions}. Use a different trigger word."
        )

    added = tok.add_tokens(placeholder_tokens)
    if added != n:
        # Defensive: should be unreachable given the pre-check, but if a
        # tokenizer implementation de-duplicates differently than we expect,
        # surface the mismatch rather than silently corrupting the TE.
        raise ValueError(
            f"Trigger {trigger!r}: pre-check passed but add_tokens returned "
            f"{added}/{n}. Tokenizer state may be inconsistent."
        )

    te.resize_token_embeddings(len(tok))
    placeholder_ids = [tok.convert_tokens_to_ids(t) for t in placeholder_tokens]

    embed_tokens = te.get_input_embeddings()
    weight = embed_tokens.weight
    vecs = pivotal.vectors.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        for i, pid in enumerate(placeholder_ids):
            weight[pid] = vecs[i]

    # Deterministic round-trip check (SC-6) — closes the silent-failure loop
    # at wire-time. atol=5e-2 accounts for fp32→bf16→fp32 precision loss: bf16
    # ULP scales with magnitude (ULP ~= |value| * 2^-8), so at the ~3-4
    # magnitudes seen in torch.randn-sized tensors the per-value error can
    # reach ~1.5e-2. Real trained embeddings are typically smaller (magnitude
    # ~0.02-1.0) with correspondingly tighter round-trip error, but the check
    # must hold under the looser worst case. A genuinely wrong row (random-init
    # vs trained) differs by ~1.0 in magnitude — 20x the bound — so this still
    # catches silent misrouting.
    #
    # Intentionally NOT `assert` — assert statements are stripped under
    # `python -O` / PYTHONOPTIMIZE=1, which would defeat the entire silent-drop
    # prevention design goal. Use explicit `raise RuntimeError` so the guard is
    # preserved under every Python invocation mode.
    trigger_id = tok.convert_tokens_to_ids(trigger)
    if trigger_id != placeholder_ids[0]:
        raise RuntimeError(
            f"pivotal round-trip: trigger id mismatch "
            f"({trigger_id} vs {placeholder_ids[0]})"
        )
    te_rows = embed_tokens.weight[placeholder_ids].detach().float().cpu()
    src = pivotal.vectors.detach().float().cpu()
    if not torch.allclose(te_rows, src, atol=5e-2):
        raise RuntimeError(
            f"pivotal round-trip: vector mismatch, max abs diff "
            f"{(te_rows - src).abs().max().item():.3e}"
        )

    logger.info(
        "Pivotal: loaded %d tokens for %r from %s",
        n,
        trigger,
        pivotal.source_path,
    )
    return placeholder_ids


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


def _patch_encode_prompt(pipe) -> None:
    """Wrap ``pipe.encode_prompt`` to expand bare triggers before tokenization.

    Instance-level monkey-patch (does not touch the class). Called once after
    ``apply_pivotal_to_pipe``. Covers all three inference paths that pass a
    string prompt through ``encode_prompt``:
      1. ``ImageEngine.generate`` → ``self._pipe(prompt=...)`` → internal encode
      2. all-on-GPU batch ``encode_and_generate`` → same
      3. 2-phase batch phase-1 ``engine.encode_prompt`` → direct

    The 4th path (``generate_from_embeddings``) passes precomputed
    ``prompt_embeds=`` and bypasses ``encode_prompt`` entirely, but it
    consumes embeddings produced by phase-1, which already went through this
    patch.
    """
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
