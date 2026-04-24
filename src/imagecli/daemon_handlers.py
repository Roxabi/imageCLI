"""Job handlers + worker loop for the imagecli daemon.

Split out of `daemon.py` to keep both files under the repo's 300 LOC gate.
Wire protocol (`_send_json`) lives in `daemon.py` and is imported here.
Model loaders live in `daemon.py` (called once at startup).
"""

from __future__ import annotations

import queue
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from imagecli.daemon import _send_json


@dataclass
class _Job:
    conn: socket.socket
    req: dict


def run_worker(q: queue.Queue, pipe: object, encoder_pipe: object) -> None:
    """Worker thread — processes encode and generate jobs sequentially, keeping models in VRAM."""
    while True:
        job: _Job = q.get()
        try:
            if job.req.get("action") == "encode":
                _handle_encode(job.conn, job.req, encoder_pipe)
            elif job.req.get("action") == "blend":
                _handle_blend(job.conn, job.req)
            else:
                _handle_job(job.conn, job.req, pipe)
        except Exception as exc:
            print(f"[imagecli daemon] worker error: {exc}", flush=True)
        finally:
            q.task_done()


def _handle_blend(conn: socket.socket, req: dict) -> None:
    """Blend pre-encoded .pt embeddings by weighted sum. No model needed."""
    import torch

    try:
        inputs = req.get("inputs")  # [{path, weight}, ...]
        out_path_str = req.get("out_path")
        if not inputs or not out_path_str:
            _send_json(conn, {"ok": False, "error": "missing inputs or out_path"})
            return

        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load all embeds and weighted-sum each tensor key
        # Work in float32 for precision, then cast back to original dtype
        blended: dict = {}
        orig_dtypes: dict = {}
        for entry in inputs:
            w = float(entry.get("weight", 1.0))
            data = torch.load(entry["path"], weights_only=True)
            for key, tensor in data.items():
                if key not in orig_dtypes:
                    orig_dtypes[key] = tensor.dtype
                t = tensor.float() * w
                blended[key] = blended[key] + t if key in blended else t

        # Restore original dtypes
        for key in blended:
            if key in orig_dtypes:
                blended[key] = blended[key].to(orig_dtypes[key])

        torch.save(blended, out_path)
        _send_json(conn, {"ok": True})

    except Exception as exc:
        try:
            _send_json(conn, {"ok": False, "error": str(exc)})
        except Exception:
            pass
    finally:
        conn.close()


def _handle_encode(conn: socket.socket, req: dict, encoder_pipe: object) -> None:
    """Encode prompts to .pt embedding files. Called exclusively from the worker thread."""
    import torch

    try:
        jobs = req.get("jobs")
        if not jobs or not isinstance(jobs, list):
            _send_json(conn, {"ok": False, "error": "missing or empty 'jobs' list"})
            return

        encoded = []
        t0 = time.time()

        for i, job in enumerate(jobs):
            job_id = job.get("id")
            prompt = job.get("prompt")
            embed_path_str = job.get("embed_path")

            if not job_id or not prompt or not embed_path_str:
                _send_json(
                    conn, {"ok": False, "error": f"job {i}: missing id, prompt, or embed_path"}
                )
                return

            embed_path = Path(embed_path_str)
            embed_path.parent.mkdir(parents=True, exist_ok=True)

            if embed_path.exists():
                progress_msg = f"[{i + 1}/{len(jobs)}] {job_id} cached"
                _send_json(conn, {"progress": progress_msg})
                encoded.append(job_id)
                continue

            negative_prompt = job.get("negative_prompt", "")

            with torch.no_grad():
                prompt_embeds, text_ids = encoder_pipe.encode_prompt(prompt=prompt)
                neg_embeds, neg_text_ids = (
                    encoder_pipe.encode_prompt(prompt=negative_prompt)
                    if negative_prompt
                    else (None, None)
                )

            payload = {
                "prompt_embeds": prompt_embeds.cpu(),
                "text_ids": text_ids.cpu() if text_ids is not None else None,
            }
            if neg_embeds is not None:
                payload["negative_prompt_embeds"] = neg_embeds.cpu()
                payload["negative_text_ids"] = (
                    neg_text_ids.cpu() if neg_text_ids is not None else None
                )
            torch.save(payload, embed_path)

            elapsed = time.time() - t0
            progress_msg = f"[{i + 1}/{len(jobs)}] {job_id} encoded  {elapsed:.0f}s"
            print(f"[imagecli daemon] {progress_msg}", flush=True)
            _send_json(conn, {"progress": progress_msg})
            encoded.append(job_id)

        _send_json(conn, {"ok": True, "encoded": encoded})

    except Exception as exc:
        try:
            _send_json(conn, {"ok": False, "error": str(exc)})
        except Exception as send_exc:
            print(f"[imagecli daemon] warning: failed to send encode error: {send_exc}", flush=True)
    finally:
        conn.close()


def _handle_job(conn: socket.socket, req: dict, pipe: object) -> None:
    """Process one generate job. Called exclusively from the worker thread."""
    import torch
    from PIL.PngImagePlugin import PngInfo

    try:
        action = req.get("action")
        if action != "generate":
            _send_json(conn, {"ok": False, "error": f"unknown action: {action!r}"})
            return

        jobs = req.get("jobs")
        if not jobs or not isinstance(jobs, list):
            _send_json(conn, {"ok": False, "error": "missing or empty 'jobs' list"})
            return

        generated = []
        t0 = time.time()

        for i, job in enumerate(jobs):
            job_id = job.get("id")
            embed_path = job.get("embed_path")
            out_path_str = job.get("out_path")

            if not job_id or not embed_path or not out_path_str:
                _send_json(
                    conn,
                    {"ok": False, "error": f"job {i}: missing id, embed_path, or out_path"},
                )
                return

            embed_path = Path(embed_path)
            out_path = Path(out_path_str)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            seed = job.get("seed", i)
            width = job.get("width", 768)
            height = job.get("height", 1024)
            steps = job.get("steps", 28)

            data = torch.load(embed_path, weights_only=True)
            prompt_embeds = data["prompt_embeds"].to("cuda")
            neg_embeds = (
                data["negative_prompt_embeds"].to("cuda")
                if "negative_prompt_embeds" in data
                else None
            )

            with torch.no_grad():
                result = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    generator=torch.Generator("cuda").manual_seed(seed),
                )

            meta = PngInfo()
            meta.add_text("seed", str(seed))
            meta.add_text("steps", str(steps))
            meta.add_text("id", job_id)
            result.images[0].save(out_path, pnginfo=meta)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            progress_msg = f"[{i + 1}/{len(jobs)}] {job_id} saved  {rate:.2f} img/s"
            print(f"[imagecli daemon] {progress_msg}", flush=True)
            _send_json(conn, {"progress": progress_msg})

            generated.append(job_id)

        _send_json(conn, {"ok": True, "generated": generated})

    except Exception as exc:
        try:
            _send_json(conn, {"ok": False, "error": str(exc)})
        except Exception as send_exc:
            print(
                f"[imagecli daemon] warning: failed to send error response: {send_exc}",
                flush=True,
            )
    finally:
        conn.close()
