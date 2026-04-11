#!/usr/bin/env python3
"""
Score images with Buffalo (face identity) and CLIP (prompt alignment).

Usage:
    uv run python scripts/score_compare.py \
        --pulid ~/.roxabi/forge/lyra/brand/concepts/avatar-lyra-v25-pulid-compare/ \
        --pulid-prompts ~/.roxabi/forge/lyra/brand/prompts/v25-pulid-compare/ \
        --lora ~/.roxabi/forge/lyra/brand/concepts/avatar-lyra-v22-lora/island0/ \
        --refs P1641 P1850 P1637 P0423 P0474 \
        --out ~/.roxabi/forge/lyra/brand/concepts/avatar-lyra-v25-pulid-compare/scores.json
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ─── Buffalo (InsightFace) ────────────────────────────────────────────────────

def load_buffalo(model_root="~/ComfyUI/models/insightface"):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="antelopev2",
        root=str(Path(model_root).expanduser()),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def face_embedding(app, img_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    # largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.normed_embedding
    return emb / np.linalg.norm(emb)


def mean_ref_embedding(app, ref_paths: list[Path]) -> np.ndarray:
    embs = []
    for p in ref_paths:
        e = face_embedding(app, p)
        if e is not None:
            embs.append(e)
    if not embs:
        raise ValueError("No faces detected in any reference image")
    mean = np.mean(embs, axis=0)
    return mean / np.linalg.norm(mean)


def buffalo_score(emb: np.ndarray | None, ref_emb: np.ndarray) -> float | None:
    if emb is None:
        return None
    return float(np.dot(emb, ref_emb))


# ─── CLIP ─────────────────────────────────────────────────────────────────────

def load_clip(model_name="ViT-L-14", pretrained="openai"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model.cuda()
    return model, preprocess, tokenizer


@torch.inference_mode()
def clip_score(model, preprocess, tokenizer, img_path: Path, prompt: str) -> float:
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
    tok = tokenizer([prompt]).cuda()
    img_feat = model.encode_image(img)
    txt_feat = model.encode_text(tok)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    return float((img_feat @ txt_feat.T).item())


# ─── Prompt parsers ───────────────────────────────────────────────────────────

def parse_md_prompt(md_path: Path) -> str:
    text = md_path.read_text()
    # strip YAML frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        return parts[2].strip() if len(parts) >= 3 else text.strip()
    return text.strip()


def parse_txt_prompt(txt_path: Path) -> str:
    text = txt_path.read_text().strip()
    # strip LoRA trigger word prefix (e.g. "lyraface ")
    text = re.sub(r"^lyraface\s+", "", text)
    return text


# ─── Main ─────────────────────────────────────────────────────────────────────

def score_set(
    label: str,
    img_dir: Path,
    prompt_source,  # Path (dir of .md) | None (use .txt next to images)
    app,
    ref_emb: np.ndarray,
    clip_model,
    preprocess,
    tokenizer,
) -> dict:
    images = sorted(img_dir.glob("*.png"))
    print(f"\n[{label}] scoring {len(images)} images …")
    results = {}
    no_face = 0
    for i, img_path in enumerate(images, 1):
        # ── prompt
        if prompt_source is not None:
            # .md file with same stem
            md = prompt_source / (img_path.stem + ".md")
            prompt = parse_md_prompt(md) if md.exists() else ""
        else:
            txt = img_dir / (img_path.stem + ".txt")
            prompt = parse_txt_prompt(txt) if txt.exists() else ""

        # ── buffalo
        emb = face_embedding(app, img_path)
        bscore = buffalo_score(emb, ref_emb)
        if emb is None:
            no_face += 1

        # ── clip
        cscore = clip_score(clip_model, preprocess, tokenizer, img_path, prompt) if prompt else None

        results[img_path.name] = {"buffalo": bscore, "clip": cscore}

        if i % 50 == 0 or i == len(images):
            b_vals = [v["buffalo"] for v in results.values() if v["buffalo"] is not None]
            c_vals = [v["clip"] for v in results.values() if v["clip"] is not None]
            print(
                f"  {i}/{len(images)}  buffalo={np.mean(b_vals):.4f}  clip={np.mean(c_vals):.4f}"
                f"  no-face={no_face}"
            )
    return results


def summary(results: dict) -> dict:
    b_vals = [v["buffalo"] for v in results.values() if v["buffalo"] is not None]
    c_vals = [v["clip"] for v in results.values() if v["clip"] is not None]
    no_face = sum(1 for v in results.values() if v["buffalo"] is None)
    return {
        "n": len(results),
        "no_face": no_face,
        "buffalo_mean": round(float(np.mean(b_vals)), 4) if b_vals else None,
        "buffalo_median": round(float(np.median(b_vals)), 4) if b_vals else None,
        "buffalo_std": round(float(np.std(b_vals)), 4) if b_vals else None,
        "clip_mean": round(float(np.mean(c_vals)), 4) if c_vals else None,
        "clip_median": round(float(np.median(c_vals)), 4) if c_vals else None,
        "clip_std": round(float(np.std(c_vals)), 4) if c_vals else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pulid", required=True, type=Path)
    parser.add_argument("--pulid-prompts", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument(
        "--refs",
        nargs="+",
        default=["P1641", "P1850", "P1637", "P0423", "P0474"],
        help="Stem prefixes of reference images in lora top30/",
    )
    parser.add_argument("--refs-dir", type=Path,
        default=Path("~/.roxabi/forge/lyra/brand/concepts/avatar-lyra-v22-lora/top30").expanduser())
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    # resolve ref images
    ref_paths = []
    for stem in args.refs:
        matches = list(args.refs_dir.glob(f"{stem}*.png"))
        if not matches:
            print(f"  ⚠ ref not found: {stem}")
        else:
            ref_paths.append(matches[0])
    print(f"Reference images ({len(ref_paths)}):")
    for p in ref_paths:
        print(f"  {p.name}")

    print("\nLoading Buffalo (antelopev2)…")
    app = load_buffalo()
    ref_emb = mean_ref_embedding(app, ref_paths)
    print(f"Reference embedding computed from {len(ref_paths)} faces.")

    print("\nLoading CLIP (ViT-L-14)…")
    clip_model, preprocess, tokenizer = load_clip()

    pulid_results = score_set(
        "PuLID", args.pulid.expanduser(), args.pulid_prompts.expanduser(),
        app, ref_emb, clip_model, preprocess, tokenizer,
    )
    lora_results = score_set(
        "LoRA", args.lora.expanduser(), None,
        app, ref_emb, clip_model, preprocess, tokenizer,
    )

    out = {
        "refs": [p.name for p in ref_paths],
        "summary": {
            "pulid": summary(pulid_results),
            "lora": summary(lora_results),
        },
        "images": {
            "pulid": pulid_results,
            "lora": lora_results,
        },
    }

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {args.out}")

    # ── print comparison table
    ps = out["summary"]["pulid"]
    ls = out["summary"]["lora"]
    print("\n┌────────────────┬─────────────────────┬─────────────────────┐")
    print("│ Metric         │ PuLID               │ LoRA                │")
    print("├────────────────┼─────────────────────┼─────────────────────┤")
    print(f"│ N images       │ {ps['n']:<19} │ {ls['n']:<19} │")
    print(f"│ No face        │ {ps['no_face']:<19} │ {ls['no_face']:<19} │")
    print(f"│ Buffalo mean   │ {ps['buffalo_mean']:<19} │ {ls['buffalo_mean']:<19} │")
    print(f"│ Buffalo median │ {ps['buffalo_median']:<19} │ {ls['buffalo_median']:<19} │")
    print(f"│ Buffalo std    │ {ps['buffalo_std']:<19} │ {ls['buffalo_std']:<19} │")
    print(f"│ CLIP mean      │ {ps['clip_mean']:<19} │ {ls['clip_mean']:<19} │")
    print(f"│ CLIP median    │ {ps['clip_median']:<19} │ {ls['clip_median']:<19} │")
    print(f"│ CLIP std       │ {ps['clip_std']:<19} │ {ls['clip_std']:<19} │")
    print("└────────────────┴─────────────────────┴─────────────────────┘")


if __name__ == "__main__":
    main()
