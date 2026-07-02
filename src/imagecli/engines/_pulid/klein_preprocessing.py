"""Identity-token preprocessing for the FLUX.2-klein PuLID engines.

Shared by :class:`PuLIDFlux2KleinEngine` and :class:`PuLIDFlux2KleinFP4Engine`
so any bug fix applies to both. Feeds the ``id_former`` of a loaded
:class:`PuLIDFlux2` with ArcFace + EVA-CLIP embeddings averaged across one or
more reference face images.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .klein_modules import PuLIDFlux2


def extract_id_tokens(
    insightface: object,
    eva_clip: object,
    pulid: PuLIDFlux2,
    face_image_paths: str | list[str],
) -> torch.Tensor:
    """Extract PuLID identity tokens from one or more face reference images.

    Multiple references are averaged in ArcFace + CLIP embedding space before
    ``id_former``, giving a centroid identity that is more pose/lighting-
    invariant than any single image. Single-image path (``str``) is accepted
    for backward compatibility.
    """
    from PIL import Image

    if isinstance(face_image_paths, str):
        face_image_paths = [face_image_paths]

    device = torch.device("cuda")
    dtype = torch.bfloat16
    mean_norm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std_norm = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    id_embeds: list[torch.Tensor] = []
    clip_embeds: list[torch.Tensor] = []

    for path in face_image_paths:
        img = np.array(Image.open(path).convert("RGB"))
        faces = insightface.get(img)  # type: ignore[union-attr]
        if not faces:
            raise RuntimeError(f"No face detected in reference image: {path}")

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        id_embed = torch.from_numpy(face.embedding).unsqueeze(0).to(device, dtype=dtype)
        id_embeds.append(F.normalize(id_embed, dim=-1))

        x1, y1, x2, y2 = face.bbox.astype(int)
        margin = int(max(x2 - x1, y2 - y1) * 0.2)
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(img.shape[1], x2 + margin), min(img.shape[0], y2 + margin)

        face_t = (
            torch.from_numpy(img[y1:y2, x1:x2].astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        face_t = F.interpolate(face_t, size=(336, 336), mode="bilinear", align_corners=False)
        face_t = (face_t - mean_norm) / std_norm

        with torch.no_grad():
            clip_out = eva_clip(face_t.float())  # type: ignore[union-attr]
            if isinstance(clip_out, (list, tuple)):
                clip_out = clip_out[0]
            if clip_out.dim() == 3:
                clip_out = clip_out[:, 0, :]
            clip_embeds.append(clip_out.to(device, dtype=dtype))

    # Average across references — valid because both spaces are L2-normalised.
    # Re-normalise the ArcFace centroid; CLIP centroid fed directly to id_former.
    id_embed = F.normalize(torch.stack(id_embeds).mean(0), dim=-1)
    clip_embed = torch.stack(clip_embeds).mean(0)

    with torch.no_grad():
        id_tokens = pulid.id_former(id_embed, clip_embed)
        id_tokens = F.normalize(id_tokens, p=2, dim=-1)

    return id_tokens
