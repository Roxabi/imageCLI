"""Engine registry + factory. Kept separate from :mod:`imagecli.engine.base` so
the ABC module doesn't transitively import every concrete engine on startup.
"""

from __future__ import annotations

import dataclasses
import warnings

from .base import ImageEngine
from imagecli.lora_spec import LoraSpec


def _get_registry() -> dict[str, type[ImageEngine]]:
    from imagecli.engines.flux1_dev import Flux1DevEngine
    from imagecli.engines.flux1_schnell import Flux1SchnellEngine
    from imagecli.engines.flux2_klein import Flux2KleinEngine
    from imagecli.engines.flux2_klein_fp4 import Flux2KleinFP4Engine
    from imagecli.engines.flux2_klein_fp8 import Flux2KleinFP8Engine
    from imagecli.engines.pulid_flux1_dev import PuLIDFlux1DevEngine
    from imagecli.engines.pulid_flux2_klein import PuLIDFlux2KleinEngine
    from imagecli.engines.pulid_flux2_klein_fp4 import PuLIDFlux2KleinFP4Engine
    from imagecli.engines.sd35 import SD35Engine

    return {
        "flux2-klein": Flux2KleinEngine,
        "flux2-klein-fp8": Flux2KleinFP8Engine,
        "flux2-klein-fp4": Flux2KleinFP4Engine,
        "pulid-flux2-klein": PuLIDFlux2KleinEngine,
        "pulid-flux2-klein-fp4": PuLIDFlux2KleinFP4Engine,
        "flux1-dev": Flux1DevEngine,
        "pulid-flux1-dev": PuLIDFlux1DevEngine,
        "flux1-schnell": Flux1SchnellEngine,
        "sd35": SD35Engine,
    }


def get_engine(
    name: str,
    *,
    compile: bool = True,
    loras: list[LoraSpec] | None = None,
    # Legacy singular kwargs — kept for backward compatibility with callers that
    # haven't migrated yet. Mutually exclusive with loras=.
    lora_path: str | None = None,
    lora_scale: float = 1.0,
    trigger: str | None = None,
    embedding_path: str | None = None,
) -> ImageEngine:
    registry = _get_registry()
    if name not in registry:
        known = ", ".join(registry)
        raise ValueError(f"Unknown engine {name!r}. Available: {known}")
    if loras is not None:
        if lora_path is not None or trigger is not None or embedding_path is not None:
            raise ValueError(
                "Pass either loras= or the singular fields "
                "(lora_path / trigger / embedding_path), not both."
            )
        return registry[name](compile=compile, loras=loras)
    # TODO(#72): remove legacy singular-kwargs branch after one release cycle.
    if lora_path is not None or trigger is not None or embedding_path is not None:
        warnings.warn(
            "get_engine singular LoRA kwargs (lora_path, lora_scale, trigger, "
            "embedding_path) are deprecated; pass loras=list[LoraSpec] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return registry[name](
        compile=compile,
        lora_path=lora_path,
        lora_scale=lora_scale,
        trigger=trigger,
        embedding_path=embedding_path,
    )


def list_engines() -> list[dict]:
    registry = _get_registry()
    return [
        {
            "name": name,
            "description": cls.description,
            "model_id": cls.model_id,
            "vram_gb": cls.vram_gb,
            "capabilities": dataclasses.asdict(cls.capabilities),
        }
        for name, cls in registry.items()
    ]
