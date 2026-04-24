"""PuLID sub-package for the FLUX.1-dev face-lock engine.

Extracted from ``pulid_flux1_dev.py`` in #55 to keep the engine file under the
300 LOC quality gate. Only the symbols re-exported here are considered public
within ``imagecli``; helper functions in ``modules`` keep their leading
underscore.
"""

from __future__ import annotations

from .modules import IDFormer, PerceiverAttention, PerceiverAttentionCA, PuLIDFlux1
from .patching import patch_flux1

__all__ = [
    "IDFormer",
    "PerceiverAttention",
    "PerceiverAttentionCA",
    "PuLIDFlux1",
    "patch_flux1",
]
