"""Markdown prompt file parser — YAML frontmatter + body text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from imagecli.lora_spec import LoraSpec

try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

_SINGULAR_LORA_KEYS = {"lora_path", "lora_scale", "trigger", "embedding_path"}


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
    face_images: list[str] | None = None
    pulid_strength: float = 0.6
    loras: list[LoraSpec] = field(default_factory=list)
    # Singular backward-compat aliases — populated from loras[0] when N==1
    lora_path: str | None = None
    lora_scale: float = 1.0
    trigger: str | None = None
    embedding_path: str | None = None
    extra: dict = field(default_factory=dict)


def _parse_loras(frontmatter: dict) -> list[LoraSpec]:
    """Extract and validate the ``loras:`` list from frontmatter.

    Raises ``ValueError`` if any item is missing ``path``.
    """
    raw = frontmatter.pop("loras")
    if not isinstance(raw, list):
        raise ValueError("Frontmatter 'loras:' must be a YAML list.")
    specs: list[LoraSpec] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"Frontmatter 'loras[{i}]' must be a mapping, got {type(item).__name__}."
            )
        if "path" not in item:
            raise ValueError(f"Frontmatter 'loras[{i}]' is missing required key 'path'.")
        specs.append(
            LoraSpec(
                path=str(item["path"]),
                scale=float(item.get("scale", 1.0)),
                trigger=item.get("trigger") or None,
                embedding_path=item.get("embedding_path") or None,
            )
        )
    return specs


def parse_prompt_file(path: Path) -> PromptDoc:
    text = path.read_text(encoding="utf-8")

    frontmatter: dict = {}
    body = text

    m = _FRONTMATTER_RE.match(text)
    if m:
        fm_text = m.group(1)
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

    # --- LoRA resolution ---
    has_list = "loras" in frontmatter
    singular_present = _SINGULAR_LORA_KEYS & frontmatter.keys()

    if has_list and singular_present:
        raise ValueError(
            "Frontmatter cannot mix 'loras:' list with singular"
            " lora_path/lora_scale/trigger/embedding_path."
        )

    loras: list[LoraSpec] = []
    if has_list:
        loras = _parse_loras(frontmatter)
    elif singular_present or "lora_path" in frontmatter:
        raw_path = frontmatter.pop("lora_path", None)
        raw_scale = _float(frontmatter.pop("lora_scale", None)) or 1.0
        raw_trigger = frontmatter.pop("trigger", None)
        raw_emb = frontmatter.pop("embedding_path", None)
        if raw_path is not None:
            loras = [
                LoraSpec(
                    path=str(raw_path), scale=raw_scale, trigger=raw_trigger, embedding_path=raw_emb
                )
            ]
    else:
        # consume singular keys even if absent so they don't leak into extra
        frontmatter.pop("lora_path", None)
        frontmatter.pop("lora_scale", None)
        frontmatter.pop("trigger", None)
        frontmatter.pop("embedding_path", None)

    # Singular backward-compat aliases
    if len(loras) == 1:
        bc_lora_path: str | None = loras[0].path
        bc_lora_scale: float = loras[0].scale
        bc_trigger: str | None = loras[0].trigger
        bc_embedding_path: str | None = loras[0].embedding_path
    else:
        bc_lora_path = None
        bc_lora_scale = 1.0
        bc_trigger = None
        bc_embedding_path = None

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
        face_images=_strlist(frontmatter.pop("face_images", None)),
        pulid_strength=_float(frontmatter.pop("pulid_strength", None)) or 0.6,
        loras=loras,
        lora_path=bc_lora_path,
        lora_scale=bc_lora_scale,
        trigger=bc_trigger,
        embedding_path=bc_embedding_path,
        extra=frontmatter,
    )


def _int(v) -> int | None:
    return int(v) if v is not None else None


def _float(v) -> float | None:
    return float(v) if v is not None else None


def _strlist(v) -> list[str] | None:
    """Accept a YAML list or a single string; return list[str] or None."""
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]
