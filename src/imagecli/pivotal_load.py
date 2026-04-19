"""Pivotal tuning embedding parsing and loading.

Parses ai-toolkit pivotal embeddings from LoRA safetensors (``emb_params``
key) or standalone A1111-format files. Returns a ``PivotalEmbedding`` ready
for ``pivotal_apply.apply_pivotal_to_pipe``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Sanity cap — pivotal tuning runs almost always use 1–8 tokens. 32 is a huge
# safety margin that still catches gibberish shapes.
_MAX_NUM_TOKENS = 32

# Upper bound on metadata["name"] raw string length before json.loads. The
# safetensors spec does not bound metadata string sizes, so a corrupted file
# could set "name" to an arbitrarily large JSON document. 256 chars is far
# larger than any realistic trigger word while keeping the parse cost trivial.
_MAX_METADATA_NAME_LEN = 256


@dataclass
class PivotalEmbedding:
    """Parsed pivotal embedding ready to apply to a Klein pipeline.

    ``vectors`` has shape ``(num_tokens, te_hidden_size)`` and is typically
    fp32 as saved by ai-toolkit. It is cast to the TE's dtype (bf16 on Klein)
    at apply time.
    """

    trigger: str
    vectors: torch.Tensor
    num_tokens: int
    source: Literal["merged", "standalone"]
    source_path: Path


def detect_pivotal_in_lora(lora_path: Path | str) -> bool:
    """Return True if the LoRA safetensors file contains an ``emb_params`` key."""
    from safetensors import safe_open

    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
        return "emb_params" in f.keys()


def _validate(tensor: torch.Tensor, te_hidden_size: int) -> None:
    """Raise ValueError if the emb_params tensor has a wrong shape."""
    ndim = tensor.ndim
    shape = tuple(tensor.shape)
    if ndim != 2:
        raise ValueError(
            f"Invalid emb_params shape {shape}: expected (N, {te_hidden_size}) "
            f"with ndim=2, got ndim={ndim}."
        )
    n, hidden = shape[0], shape[1]
    if n < 1:
        raise ValueError(
            f"Invalid emb_params shape {shape}: expected N >= 1, got N={n}."
        )
    if n > _MAX_NUM_TOKENS:
        raise ValueError(
            f"Invalid emb_params shape {shape}: expected N <= {_MAX_NUM_TOKENS} "
            f"(sanity cap), got N={n}."
        )
    if hidden != te_hidden_size:
        raise ValueError(
            f"Pivotal embedding dim ({hidden}) does not match text encoder "
            f"hidden_size ({te_hidden_size}). LoRA was likely trained against a "
            f"different base model."
        )


def load_pivotal_embedding(
    lora_path: Path | str | None,
    trigger: str | None,
    *,
    embedding_path: Path | str | None = None,
    te_hidden_size: int = 2560,
) -> PivotalEmbedding | None:
    """Resolve and load a pivotal embedding from a LoRA or standalone file.

    Resolution order:
      1. Explicit ``embedding_path`` → standalone format. Trigger may be
         inferred from metadata ``name`` if the caller did not supply one.
      2. ``lora_path`` contains ``emb_params`` → merged format.
      3. Neither → return ``None`` (caller may log + continue without pivotal).

    Raises:
      ValueError: if pivotal embedding is found but no trigger was resolved,
        or if the emb_params tensor has an invalid shape. The error message
        always mentions ``--trigger`` because silent-continue was the V23b
        failure mode this feature exists to prevent.
    """
    from safetensors import safe_open

    source: Literal["merged", "standalone"]
    source_path: Path

    # Resolve the source file and read emb_params + metadata in one pass.
    if embedding_path is not None:
        source_path = Path(embedding_path)
        source = "standalone"
    elif lora_path is not None and detect_pivotal_in_lora(lora_path):
        source_path = Path(lora_path)
        source = "merged"
    else:
        # No pivotal embedding present. Caller decides whether to warn.
        return None

    with safe_open(str(source_path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        if "emb_params" not in f.keys():
            raise KeyError(f"emb_params not found in {source_path}")
        tensor = f.get_tensor("emb_params")

    # Standalone format: trigger may be inferred from metadata "name".
    # Bound the raw metadata string before parsing — safetensors does not cap
    # metadata string sizes, so a corrupted file could carry an enormous JSON
    # document that would balloon memory via json.loads.
    if source == "standalone" and trigger is None and "name" in metadata:
        raw_name = metadata["name"]
        if len(raw_name) > _MAX_METADATA_NAME_LEN:
            raise ValueError(
                f"metadata['name'] exceeds {_MAX_METADATA_NAME_LEN} chars "
                f"({len(raw_name)}); refusing to parse as JSON."
            )
        try:
            parsed = json.loads(raw_name)
        except (json.JSONDecodeError, TypeError):
            parsed = raw_name  # raw string fallback
        if isinstance(parsed, str):
            trigger = parsed

    if trigger is None:
        raise ValueError(
            "LoRA contains emb_params (pivotal tuning) but no trigger was "
            "provided. Pass --trigger <word> or set trigger: in frontmatter. "
            "Without a trigger the embeddings are silently ignored."
        )

    _validate(tensor, te_hidden_size)
    num_tokens = int(tensor.shape[0])

    return PivotalEmbedding(
        trigger=trigger,
        vectors=tensor,
        num_tokens=num_tokens,
        source=source,
        source_path=source_path,
    )
