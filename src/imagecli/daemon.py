"""Daemon server, wire protocol, and startup model loaders for FLUX.2-klein-4B.

Protocol: newline-delimited JSON over AF_UNIX SOCK_STREAM.
Socket path: ~/.local/share/imagecli/daemon.sock

Actions:
  ping     — liveness check
  generate — generate images from pre-encoded embedding .pt files
  encode   — encode prompts to .pt files
  blend    — weighted-sum pre-encoded embeddings
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
from pathlib import Path

SOCKET_PATH = Path.home() / ".local" / "share" / "imagecli" / "daemon.sock"
_DEFAULT_TIMEOUT = 600  # seconds — large batches can take time

# Minimum free VRAM required before loading each component (GB)
_VRAM_TRANSFORMER_VAE = 4.5
_VRAM_TEXT_ENCODER = 8.0


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
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line:
                continue
            obj = json.loads(line)
            if "ok" in obj:
                return obj
            if "progress" in obj:
                print(obj["progress"], flush=True)
    return {"ok": False, "error": "daemon closed connection unexpectedly"}


# ── Server ────────────────────────────────────────────────────────────────────


def daemon_main(engine: str = "flux2-klein") -> None:
    """Start the daemon, preload transformer+VAE + text encoder, serve forever.

    Args:
        engine: Engine name (currently only 'flux2-klein' is supported).
    """
    from imagecli.daemon_handlers import _Job, run_worker

    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)

    print(f"[imagecli daemon] Preloading {engine} transformer+VAE...", flush=True)
    try:
        pipe = load_pipe()
    except RuntimeError as exc:
        print(f"[imagecli daemon] ERROR: {exc}", flush=True)
        raise SystemExit(1)

    print("[imagecli daemon] Preloading text encoder...", flush=True)
    try:
        encoder_pipe = load_encoder()
    except RuntimeError as exc:
        print(f"[imagecli daemon] ERROR: {exc}", flush=True)
        raise SystemExit(1)

    _queue: queue.Queue = queue.Queue()
    threading.Thread(target=run_worker, args=(_queue, pipe, encoder_pipe), daemon=True).start()

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


# ── Model loaders (called once from daemon_main at startup) ───────────────────


def _check_vram(needed_gb: float, label: str) -> None:
    """Raise RuntimeError if free VRAM < needed_gb. empty_cache() first to release reserved mem."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free / 1024**3, total / 1024**3
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


def load_pipe():
    """Load FLUX.2-klein-4B transformer (fp8 quantized) + VAE. Deferred heavy imports."""
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


def load_encoder():
    """Load FLUX.2-klein-4B text encoder (Qwen3) only. Deferred heavy imports."""
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
