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
    """Start the generation daemon, preloading the transformer+VAE at startup.

    Args:
        engine: Engine name (currently only 'flux2-klein' is supported).
    """
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)

    print(f"[imagecli daemon] Preloading {engine} transformer+VAE...", flush=True)
    pipe = _load_pipe()

    _queue: queue.Queue = queue.Queue()
    threading.Thread(target=_worker, args=(_queue, pipe), daemon=True).start()

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


def _load_pipe():
    """Load FLUX.2-klein-4B transformer (fp8 quantized) + VAE. Deferred import."""
    import torch
    from diffusers import Flux2KleinPipeline
    from optimum.quanto import freeze, qfloat8, quantize
    from optimum.quanto.nn import QLinear

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
    print(f"[imagecli daemon] Transformer (FP8) + VAE on GPU  VRAM: {used:.1f}/{total:.1f} GB", flush=True)

    return pipe


def _worker(q: queue.Queue, pipe: object) -> None:
    """Worker thread — processes generation jobs sequentially, keeping pipe in VRAM."""
    while True:
        job: _Job = q.get()
        try:
            _handle_job(job.conn, job.req, pipe)
        except Exception as exc:
            print(f"[imagecli daemon] worker error: {exc}", flush=True)
        finally:
            q.task_done()


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

            with torch.no_grad():
                result = pipe(
                    prompt_embeds=prompt_embeds,
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
