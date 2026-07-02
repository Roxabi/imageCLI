"""Pivotal tuning embedding loader + applier for Klein 4B (Qwen3 TE).

Thin re-export shim. Implementation split across:
- :mod:`imagecli.pivotal_load` — parsing, validation, loader
- :mod:`imagecli.pivotal_apply` — TE/tokenizer wiring and ``encode_prompt`` patch

Why this exists: imageCLI's LoRA load path fuses the transformer adapter but
never touches the tokenizer or text encoder. An ``emb_params`` tensor written
alongside the LoRA by ai-toolkit's pivotal tuning block was silently discarded
at inference — the trained TE-side contribution was dead weight. This module
closes that failure mode.

See:
- ``artifacts/analyses/31-pivotal-tuning-embeddings-analysis.mdx`` for the
  verification trail (Qwen2Tokenizer + Qwen3 TE support the standard HF API,
  no porting needed).
- ``artifacts/specs/31-pivotal-tuning-embeddings-spec.mdx`` for acceptance.
"""

from __future__ import annotations

from imagecli.pivotal_apply import (
    _maybe_convert_prompt,
    _patch_encode_prompt,
    apply_pivotal_to_pipe,
    apply_pivotals_to_pipe,
)
from imagecli.pivotal_load import (
    PivotalEmbedding,
    detect_pivotal_in_lora,
    load_pivotal_embedding,
)

__all__ = [
    "PivotalEmbedding",
    "_maybe_convert_prompt",
    "_patch_encode_prompt",
    "apply_pivotal_to_pipe",
    "apply_pivotals_to_pipe",
    "detect_pivotal_in_lora",
    "load_pivotal_embedding",
]
