"""Daemon server and client for keeping FLUX.2-klein-4B transformer+VAE warm in VRAM.

Protocol: newline-delimited JSON over AF_UNIX SOCK_STREAM.
Socket path: ~/.local/share/imagecli/daemon.sock

Actions:
  ping     — liveness check
  generate — generate images from pre-encoded embedding .pt files
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SOCKET_PATH = Path.home() / ".local" / "share" / "imagecli" / "daemon.sock"
_DEFAULT_TIMEOUT = 600  # seconds — large batches can take time


@dataclass
class _Job:
    conn: socket.socket
    req: dict


# ── Public client API ─────────────────────────────────────────────────────────


def daemon_request(request: dict, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Send a JSON request to the daemon and return the response dict.

    For 'generate' actions, reads multiple lines until one has an 'ok' key.
    Progress lines (containing 'progress' key) are printed to stdout.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(SOCKET_PATH))
        _send_json(sock, request)
        if request.get("action") == "generate":
            return _recv_generate(sock)
        return _recv_json(sock)
    finally:
        sock.close()


def _recv_generate(sock: socket.socket) -> dict:
    """Read lines until we get one with 'ok' key; print progress lines to stdout."""
    buf = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
        # Process all complete lines in buffer
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line:
                continue
            obj = json.loads(line)
            if "ok" in obj:
                return obj
            if "progress" in obj:
                print(obj["progress"], flush=True)
    # EOF without final response
    return {"ok": False, "error": "daemon closed connection unexpectedly"}


# ── Server ────────────────────────────────────────────────────────────────────


def daemon_main(engine: str = "flux2-klein") -> None:
    """Start the generation daemon, preloading transformer+VAE and text encoder at startup.

    Args:
        engine: Engine name (currently only 'flux2-klein' is supported).
    """
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)

    print(f"[imagecli daemon] Preloading {engine} transformer+VAE...", flush=True)
    try:
        pipe = _load_pipe()
    except RuntimeError as exc:
        print(f"[imagecli daemon] ERROR: {exc}", flush=True)
        raise SystemExit(1)

    print("[imagecli daemon] Preloading text encoder...", flush=True)
    try:
        encoder_pipe = _load_encoder()
    except RuntimeError as exc:
        print(f"[imagecli daemon] ERROR: {exc}", flush=True)
        raise SystemExit(1)

    _queue: queue.Queue = queue.Queue()
    threading.Thread(target=_worker, args=(_queue, pipe, encoder_pipe), daemon=True).start()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
        srv.bind(str(SOCKET_PATH))
        os.chmod(str(SOCKET_PATH), 0o600)
        srv.listen(5)
        print(f"[imagecli daemon] Ready on {SOCKET_PATH}", flush=True)
        try:
            while True:
                conn, _ = srv.accept()
                conn.settimeout(5)
                try:
                    req = _recv_json(conn)
                except Exception:
                    conn.close()
                    continue
                if req.get("action") == "ping":
                    _send_json(conn, {"ok": True})
                    conn.close()
                else:
                    # conn ownership transfers to worker — main thread must not touch conn after this
                    _queue.put(_Job(conn=conn, req=req))
        except KeyboardInterrupt:
            pass
        finally:
            SOCKET_PATH.unlink(missing_ok=True)


# ── Worker ────────────────────────────────────────────────────────────────────


# Minimum free VRAM required before loading each component (GB)
_VRAM_TRANSFORMER_VAE = 4.5
_VRAM_TEXT_ENCODER = 8.0


def _check_vram(needed_gb: float, label: str) -> None:
    """Raise RuntimeError if free VRAM is insufficient.

    Calls empty_cache() first to release PyTorch reserved-but-unallocated memory
    (e.g. leftover from a previous daemon instance that just exited).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    free_gb = free / 1024**3
    total_gb = total / 1024**3
    if free_gb < needed_gb:
        raise RuntimeError(
            f"[imagecli daemon] Not enough VRAM for {label}: "
            f"need ~{needed_gb:.1f} GB, only {free_gb:.1f}/{total_gb:.1f} GB free"
        )
    print(
        f"[imagecli daemon] VRAM check OK for {label}: "
        f"{free_gb:.1f}/{total_gb:.1f} GB free (need {needed_gb:.1f} GB)",
        flush=True,
    )


def _load_pipe():
    """Load FLUX.2-klein-4B transformer (fp8 quantized) + VAE. Deferred import."""
    import torch
    from diffusers import Flux2KleinPipeline
    from optimum.quanto import freeze, qfloat8, quantize
    from optimum.quanto.nn import QLinear

    _check_vram(_VRAM_TRANSFORMER_VAE, "transformer+VAE")

    MODEL = "black-forest-labs/FLUX.2-klein-4B"

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
    )
    quantize(pipe.transformer, weights=qfloat8)
    freeze(pipe.transformer)

    # Fix contiguity issue with QLinear on Blackwell
    _orig = QLinear.forward

    def _cont(self, inp):  # noqa: ANN001, ANN202
        return _orig(self, inp.contiguous())

    QLinear.forward = _cont

    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")

    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(
        f"[imagecli daemon] Transformer (FP8) + VAE on GPU  VRAM: {used:.1f}/{total:.1f} GB",
        flush=True,
    )

    return pipe


def _load_encoder():
    """Load FLUX.2-klein-4B text encoder (Qwen3) only. Deferred import."""
    import torch
    from diffusers import Flux2KleinPipeline

    _check_vram(_VRAM_TEXT_ENCODER, "text encoder")

    MODEL = "black-forest-labs/FLUX.2-klein-4B"

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL, transformer=None, vae=None, torch_dtype=torch.bfloat16
    )
    pipe.text_encoder.to("cuda")

    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[imagecli daemon] Text encoder on GPU  VRAM: {used:.1f}/{total:.1f} GB", flush=True)

    return pipe


def _worker(q: queue.Queue, pipe: object, encoder_pipe: object) -> None:
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


# ── Wire protocol ─────────────────────────────────────────────────────────────


def _send_json(sock: socket.socket, data: dict) -> None:
    payload = json.dumps(data, ensure_ascii=False) + "\n"
    sock.sendall(payload.encode())


def _recv_json(sock: socket.socket) -> dict:
    buf = bytearray()
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in buf:
            break
    line = buf.split(b"\n")[0]
    return json.loads(line)
