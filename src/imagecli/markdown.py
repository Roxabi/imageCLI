"""Markdown prompt file parser — YAML frontmatter + body text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class PromptDoc:
    prompt: str
    negative_prompt: str = ""
    engine: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    format: str | None = None
    face_image: str | None = None
    pulid_strength: float = 0.6
    lora_path: str | None = None
    lora_scale: float = 1.0
    extra: dict = field(default_factory=dict)


def parse_prompt_file(path: Path) -> PromptDoc:
    text = path.read_text(encoding="utf-8")

    frontmatter: dict = {}
    body = text

    m = _FRONTMATTER_RE.match(text)
    if m:
        fm_text = m.group(1)
        body = text[m.end()].strip() if m.end() < len(text) else ""
        body = text[m.end() :].strip()
        if _HAS_YAML:
            frontmatter = yaml.safe_load(fm_text) or {}
        else:
            # fallback: try key: value pairs
            for line in fm_text.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    frontmatter[k.strip()] = v.strip().strip("\"'")

    prompt = frontmatter.pop("prompt", body).strip()

    return PromptDoc(
        prompt=prompt,
        negative_prompt=str(frontmatter.pop("negative_prompt", "")),
        engine=frontmatter.pop("engine", None),
        width=_int(frontmatter.pop("width", None)),
        height=_int(frontmatter.pop("height", None)),
        steps=_int(frontmatter.pop("steps", None)),
        guidance=_float(frontmatter.pop("guidance", None)),
        seed=_int(frontmatter.pop("seed", None)),
        format=frontmatter.pop("format", None),
        face_image=frontmatter.pop("face_image", None),
        pulid_strength=_float(frontmatter.pop("pulid_strength", None)) or 0.6,
        lora_path=frontmatter.pop("lora_path", None),
        lora_scale=_float(frontmatter.pop("lora_scale", None)) or 1.0,
        extra=frontmatter,
    )


def _int(v) -> int | None:
    return int(v) if v is not None else None


def _float(v) -> float | None:
    return float(v) if v is not None else None
