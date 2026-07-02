"""PuLID sub-package for the FLUX PuLID face-lock engines.

Extracted from the monolithic engine files in #55 (FLUX.1-dev) and #56
(FLUX.2-klein) to keep engine files under the 300 LOC quality gate. Only
the symbols re-exported here are considered public within ``imagecli``;
helper functions in the submodules keep their leading underscore.
"""

from __future__ import annotations

from .klein_modules import KleinIDFormer, KleinPerceiverAttentionCA, PuLIDFlux2
from .klein_patching import patch_flux2
from .klein_preprocessing import extract_id_tokens
from .modules import IDFormer, PerceiverAttention, PerceiverAttentionCA, PuLIDFlux1
from .patching import patch_flux1

__all__ = [
    "IDFormer",
    "KleinIDFormer",
    "KleinPerceiverAttentionCA",
    "PerceiverAttention",
    "PerceiverAttentionCA",
    "PuLIDFlux1",
    "PuLIDFlux2",
    "extract_id_tokens",
    "patch_flux1",
    "patch_flux2",
]
