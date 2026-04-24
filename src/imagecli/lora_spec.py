"""Canonical internal LoRA specification.

One :class:`LoraSpec` per LoRA adapter that an :class:`ImageEngine` loads.
The singular legacy kwargs (``lora_path``, ``lora_scale``, ``trigger``,
``embedding_path``) on engine ``__init__`` fold into a one-element
``list[LoraSpec]`` for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

# Practical cap on stack depth. Not a security-critical bound (the CLI runs
# locally and the NATS bus is on a trusted LAN), but prevents adversarial or
# accidental N≫ N (e.g. a misformed YAML list, a generator loop) from OOMing
# the daemon while holding the concurrency semaphore. Kept liberal (8) so
# legitimate experimentation with style+subject+face+refiner stacks fits.
MAX_LORAS = 8


@dataclass(frozen=True)
class LoraSpec:
    """A single LoRA adapter, optionally with a pivotal tuning embedding.

    ``path`` is the on-disk location of the ``.safetensors`` adapter.
    ``scale`` is the adapter weight at fuse time (1.0 = unchanged).
    ``trigger`` is the bare trigger word the TE embedding rows were trained
    against (required only if ``path`` contains ``emb_params`` or if
    ``embedding_path`` is set and its metadata does not carry a name).
    ``embedding_path`` is an explicit standalone pivotal embedding file;
    when set it overrides any ``emb_params`` present in ``path``.
    """

    path: str
    scale: float = 1.0
    trigger: str | None = None
    embedding_path: str | None = None
