"""Abstract ImageEngine base class + engine registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ImageEngine(ABC):
    name: str
    description: str
    model_id: str
    vram_gb: float  # approximate minimum VRAM
    default_steps: int = 50
    default_guidance: float = 4.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        guidance: float = 4.0,
        seed: int | None = None,
        output_path: Path,
        **kwargs,
    ) -> Path:
        """Generate an image and save it to output_path. Returns the saved path."""


def _get_registry() -> dict[str, type[ImageEngine]]:
    from imagecli.engines.flux2_klein import Flux2KleinEngine
    from imagecli.engines.flux1_dev import Flux1DevEngine
    from imagecli.engines.sd35 import SD35Engine

    return {
        "flux2-klein": Flux2KleinEngine,
        "flux1-dev": Flux1DevEngine,
        "sd35": SD35Engine,
    }


def get_engine_class(name: str) -> type[ImageEngine]:
    registry = _get_registry()
    if name not in registry:
        known = ", ".join(registry)
        raise ValueError(f"Unknown engine {name!r}. Available: {known}")
    return registry[name]


def get_engine(name: str) -> ImageEngine:
    registry = _get_registry()
    if name not in registry:
        known = ", ".join(registry)
        raise ValueError(f"Unknown engine {name!r}. Available: {known}")
    return registry[name]()


def list_engines() -> list[dict]:
    registry = _get_registry()
    return [
        {
            "name": name,
            "description": cls.description,
            "model_id": cls.model_id,
            "vram_gb": cls.vram_gb,
            "default_steps": cls.default_steps,
            "default_guidance": cls.default_guidance,
        }
        for name, cls in registry.items()
    ]
