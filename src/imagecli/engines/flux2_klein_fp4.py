"""FLUX.2-klein-4B NVFP4 engine — Blackwell-native FP4 via comfy-kitchen.

Uses BFL official NVFP4 weights with comfy-kitchen's scaled_mm_nvfp4 kernel.
~2 GB transformer, ~11.5 GB all-on-GPU, ~6.8 it/s at 512x512 on RTX 5070 Ti.

LoRA support: when --lora is set, loads bf16 base, fuses LoRA at --lora-scale,
then runtime-quantizes the fused transformer to NVFP4 via
QuantizedTensor.from_float(...). Pre-quantized disk weights are bypassed in this
path (they encode the un-fused model). V22 validation: NVFP4 + fused LoRA scores
−0.032 vs FP8 + fused LoRA, for 8× speedup and 60% VRAM reduction.

Requires Blackwell GPU (sm_120+), CUDA 13.0 (cu130), and comfy-kitchen[cublas].

Install: uv sync --group fp4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from imagecli.engine import EngineCapabilities, ImageEngine

logger = logging.getLogger(__name__)

NVFP4_REPO = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
NVFP4_FILENAME = "flux-2-klein-4b-nvfp4.safetensors"
BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"

# ── BFL → diffusers key mapping (same as diffusers single_file_utils.py) ─────

_RENAME = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
    "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
    "single_stream_modulation.lin": "single_stream_modulation.linear",
    "final_layer.linear": "proj_out",
}

_DOUBLE_BLOCK_MAP = {
    "img_attn.norm.query_norm": "attn.norm_q",
    "img_attn.norm.key_norm": "attn.norm_k",
    "img_attn.proj": "attn.to_out.0",
    "img_mlp.0": "ff.linear_in",
    "img_mlp.2": "ff.linear_out",
    "txt_attn.norm.query_norm": "attn.norm_added_q",
    "txt_attn.norm.key_norm": "attn.norm_added_k",
    "txt_attn.proj": "attn.to_add_out",
    "txt_mlp.0": "ff_context.linear_in",
    "txt_mlp.2": "ff_context.linear_out",
}

_SINGLE_BLOCK_MAP = {
    "linear1": "attn.to_qkv_mlp_proj",
    "linear2": "attn.to_out",
    "norm.query_norm": "attn.norm_q",
    "norm.key_norm": "attn.norm_k",
}

# NVFP4 tensor suffixes: each linear layer has these 4 tensors
_NVFP4_SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")


def _convert_nvfp4_state_dict(raw: dict[str, object]) -> dict[str, object]:
    """Convert BFL NVFP4 state dict keys to diffusers naming, splitting fused QKV."""
    import torch

    out: dict[str, object] = {}

    # Step 1: Simple renames
    for old_key in list(raw.keys()):
        new_key = old_key
        for old, new in _RENAME.items():
            new_key = new_key.replace(old, new)
        if new_key != old_key:
            raw[new_key] = raw.pop(old_key)

    # Step 2: Double blocks — handle QKV split and non-QKV remap
    for key in list(raw.keys()):
        if "double_blocks." not in key:
            continue

        parts = key.split(".")
        block_idx = parts[1]
        # Reconstruct the within-block name (everything between block idx and the last part)
        # e.g. "double_blocks.0.img_attn.qkv.weight" → within="img_attn.qkv", suffix=".weight"
        rest = ".".join(parts[2:])

        # Check for NVFP4 suffix
        suffix = None
        for s in _NVFP4_SUFFIXES + (".scale",):
            if rest.endswith(s.lstrip(".")):
                suffix = rest[rest.rindex(s.lstrip(".")) - 1:]  # include the dot
                within = rest[: rest.rindex(s.lstrip(".")) - 1]
                break
        if suffix is None:
            # Regular weight/bias
            within = ".".join(parts[2:-1])
            suffix = "." + parts[-1]

        if suffix == ".scale":
            suffix = ".weight"

        if "qkv" in within:
            # Fused QKV → split into Q, K, V (split dim=0 into 3 chunks)
            tensor = raw.pop(key)
            if tensor.dim() >= 1:
                chunks = torch.chunk(tensor, 3, dim=0)
            else:
                # Scalar (e.g. weight_scale_2, input_scale) — duplicate for each
                chunks = (tensor, tensor, tensor)

            if "img" in within:
                names = ("attn.to_q", "attn.to_k", "attn.to_v")
            else:
                names = ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")

            for name, chunk in zip(names, chunks):
                out_key = f"transformer_blocks.{block_idx}.{name}{suffix}"
                out[out_key] = chunk
        else:
            # Non-QKV — simple remap
            if within in _DOUBLE_BLOCK_MAP:
                new_within = _DOUBLE_BLOCK_MAP[within]
                out_key = f"transformer_blocks.{block_idx}.{new_within}{suffix}"
                out[out_key] = raw.pop(key)

    # Step 3: Single blocks — simple remap
    for key in list(raw.keys()):
        if "single_blocks." not in key:
            continue

        parts = key.split(".")
        block_idx = parts[1]
        rest = ".".join(parts[2:])

        suffix = None
        for s in _NVFP4_SUFFIXES + (".scale",):
            if rest.endswith(s.lstrip(".")):
                suffix = rest[rest.rindex(s.lstrip(".")) - 1:]
                within = rest[: rest.rindex(s.lstrip(".")) - 1]
                break
        if suffix is None:
            within = ".".join(parts[2:-1])
            suffix = "." + parts[-1]

        if suffix == ".scale":
            suffix = ".weight"

        if within in _SINGLE_BLOCK_MAP:
            new_within = _SINGLE_BLOCK_MAP[within]
            out_key = f"single_transformer_blocks.{block_idx}.{new_within}{suffix}"
            out[out_key] = raw.pop(key)

    # Step 4: Anything remaining (already renamed in step 1)
    for key, val in raw.items():
        if key not in out:
            out[key] = val

    return out


def _runtime_quantize_transformer_to_nvfp4(transformer) -> int:
    """Runtime-quantize all nn.Linear weights in the transformer to NVFP4.

    Used when a LoRA has been fused into the bf16 base weights — we cannot use
    the pre-quantized disk weights because they encode the un-fused model.
    Instead we quantize the (LoRA-fused) bf16 weights on the fly via
    QuantizedTensor.from_float(w, "TensorCoreNVFP4Layout").

    Returns the number of layers quantized.
    """
    import torch
    from comfy_kitchen.tensor import QuantizedTensor

    quantized = 0
    for module in transformer.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        w = module.weight.data
        if w.dtype != torch.bfloat16:
            continue  # already quantized or wrong dtype
        qt = QuantizedTensor.from_float(
            w.to("cuda"), layout_cls="TensorCoreNVFP4Layout"
        )
        module.weight = torch.nn.Parameter(qt, requires_grad=False)
        quantized += 1
    return quantized


def _patch_transformer_nvfp4(transformer, nvfp4_path: str) -> int:
    """Replace linear layer weights with NVFP4 QuantizedTensors."""
    import torch
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as NVFP4
    from safetensors import safe_open

    # Load and convert BFL keys to diffusers keys
    with safe_open(nvfp4_path, framework="pt", device="cuda") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}

    converted = _convert_nvfp4_state_dict(raw)

    # Group by layer: find layers that have .weight (uint8) + .weight_scale + .weight_scale_2
    layers: dict[str, dict[str, object]] = {}
    for key, tensor in converted.items():
        for suffix in _NVFP4_SUFFIXES:
            if key.endswith(suffix):
                layer_name = key[: -len(suffix)]
                layers.setdefault(layer_name, {})[suffix] = tensor
                break

    patched = 0
    for layer_name, tensors in layers.items():
        if ".weight" not in tensors or ".weight_scale" not in tensors:
            continue

        weight = tensors[".weight"]
        if weight.dtype != torch.uint8:
            continue  # Not a quantized layer

        block_scale = tensors[".weight_scale"]
        global_scale = tensors.get(".weight_scale_2", torch.tensor(1.0, dtype=torch.float32))

        orig_shape = (weight.shape[0], weight.shape[1] * 2)
        params = NVFP4.Params(
            scale=global_scale.to("cuda"),
            orig_dtype=torch.bfloat16,
            orig_shape=orig_shape,
            block_scale=block_scale.to("cuda"),
        )
        qt = QuantizedTensor(weight.to("cuda"), layout_cls="TensorCoreNVFP4Layout", params=params)

        # Find the module and replace its weight
        parts = layer_name.split(".")
        module = transformer
        for part in parts:
            module = getattr(module, part, None)
            if module is None:
                break

        if module is not None and isinstance(module, torch.nn.Linear):
            module.weight = torch.nn.Parameter(qt, requires_grad=False)
            patched += 1

    # Also load non-quantized weights (norm scales, biases)
    for key, tensor in converted.items():
        is_nvfp4 = any(key.endswith(s) for s in _NVFP4_SUFFIXES)
        if is_nvfp4:
            continue
        # Try to set this parameter on the transformer
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            continue
        module_path, param_name = parts
        module = transformer
        for part in module_path.split("."):
            module = getattr(module, part, None)
            if module is None:
                break
        if module is not None and hasattr(module, param_name):
            param = getattr(module, param_name)
            if isinstance(param, torch.nn.Parameter):
                param.data.copy_(tensor.to(param.device))
            elif isinstance(param, torch.Tensor):
                param.copy_(tensor.to(param.device))

    return patched


class Flux2KleinFP4Engine(ImageEngine):
    name = "flux2-klein-fp4"
    description = "FLUX.2-klein-4B NVFP4 — Blackwell FP4 via comfy-kitchen (~2 GB transformer)"
    model_id = BASE_REPO
    vram_gb = 4.0
    capabilities = EngineCapabilities(negative_prompt=False)
    supports_two_phase: ClassVar[bool] = True

    def _check_requirements(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("NVFP4 requires a CUDA GPU.")
        sm = torch.cuda.get_device_capability()
        if sm < (12, 0):
            raise RuntimeError(
                f"NVFP4 requires Blackwell GPU (sm_120+), got sm_{sm[0]}{sm[1]}."
            )
        cuda_ver = torch.version.cuda
        if cuda_ver and int(cuda_ver.split(".")[0]) < 13:
            raise RuntimeError(f"NVFP4 requires CUDA 13.0+ (cu130), got {cuda_ver}.")
        try:
            import comfy_kitchen  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "flux2-klein-fp4 requires comfy-kitchen. Install: uv sync --group fp4"
            )

    def _load_pipeline(self):
        if self._pipe is not None:
            return

        self._check_requirements()

        import torch
        from diffusers import Flux2KleinPipeline
        from huggingface_hub import hf_hub_download

        logger.info("Loading base pipeline %s...", BASE_REPO)
        self._pipe = Flux2KleinPipeline.from_pretrained(
            BASE_REPO, torch_dtype=torch.bfloat16
        )

        if self.lora_path:
            # LoRA path: fuse LoRA into bf16 base, then runtime-quantize the fused
            # weights to NVFP4. Pre-quantized disk weights would overwrite the
            # fused delta, so we bypass them entirely.
            logger.info("Loading LoRA from %s...", self.lora_path)
            self._pipe.load_lora_weights(self.lora_path)
            if self.lora_scale != 1.0:
                self._pipe.set_adapters(["default_0"], adapter_weights=[self.lora_scale])
                logger.info("LoRA adapter scale set to %.2f", self.lora_scale)
            self._pipe.fuse_lora()
            self._pipe.unload_lora_weights()
            logger.info("LoRA fused into bf16 base weights.")

            # Pivotal tuning: load trigger embeddings into the TE BEFORE moving
            # the transformer to GPU. TE is still on CPU in bf16 — disjoint from
            # runtime NVFP4 quantization (which only touches nn.Linear layers
            # in the transformer, not nn.Embedding in the TE).
            self._apply_pivotal_if_needed()

            self._pipe.transformer.to("cuda")
            logger.info("Runtime-quantizing fused transformer to NVFP4...")
            patched = _runtime_quantize_transformer_to_nvfp4(self._pipe.transformer)
            logger.info("Runtime-quantized %d linear layers to NVFP4.", patched)
        else:
            # Pivotal-only path: standalone embedding file without a LoRA.
            if self.embedding_path:
                self._apply_pivotal_if_needed()

            # No LoRA: use pre-quantized BFL NVFP4 weights from the hub.
            logger.info("Downloading NVFP4 weights from %s...", NVFP4_REPO)
            nvfp4_path = hf_hub_download(repo_id=NVFP4_REPO, filename=NVFP4_FILENAME)

            self._pipe.transformer.to("cuda")
            logger.info("Patching transformer with NVFP4 QuantizedTensors...")
            patched = _patch_transformer_nvfp4(self._pipe.transformer, nvfp4_path)
            logger.info("Patched %d linear layers with NVFP4 weights.", patched)

    def _apply_pivotal_if_needed(self):
        """Load pivotal embeddings into the TE if a trigger/embedding is set."""
        if not (self.lora_path or self.embedding_path):
            return
        from imagecli.pivotal import (
            _patch_encode_prompt,
            apply_pivotal_to_pipe,
            load_pivotal_embedding,
        )

        pivotal = load_pivotal_embedding(
            self.lora_path,
            self.trigger,
            embedding_path=self.embedding_path,
            te_hidden_size=self._pipe.text_encoder.config.hidden_size,
        )
        if pivotal is not None:
            apply_pivotal_to_pipe(self._pipe, pivotal)
            _patch_encode_prompt(self._pipe)

    def _set_execution_device(self):
        import torch

        self._pipe._execution_device_override = torch.device("cuda")
        orig_cls = type(self._pipe)
        if not hasattr(orig_cls, "_orig_execution_device"):
            orig_cls._orig_execution_device = orig_cls._execution_device
            orig_cls._execution_device = property(
                lambda pipe: getattr(pipe, "_execution_device_override", None)
                or orig_cls._orig_execution_device.fget(pipe)
            )

    def _load(self):
        if self._pipe is not None:
            return
        self._load_pipeline()
        self._pipe.vae.to("cuda")
        self._pipe.text_encoder.to("cuda")
        self._optimize_pipe(self._pipe, compile=False)
        self._set_execution_device()
        logger.info("Model ready (all on GPU, NVFP4 transformer).")

    # ── All-on-GPU batch ─────────────────────────────────────────────────

    def load_all_on_gpu(self):
        self._load_pipeline()
        self._pipe.text_encoder.to("cuda")
        self._pipe.vae.to("cuda")
        self._set_execution_device()
        self._optimize_pipe(self._pipe, compile=False)
        logger.info("All components on GPU — NVFP4 transformer.")

    def encode_and_generate(
        self, prompt: str, *, width: int = 1024, height: int = 1024,
        steps: int = 50, guidance: float = 4.0, seed: int | None = None,
        output_path: Path, callback=None,
    ) -> Path:
        import random
        import torch

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        pipe_kwargs = {
            "prompt": prompt, "width": width, "height": height,
            "num_inference_steps": steps, "guidance_scale": guidance,
            "generator": generator,
        }
        if callback is not None:
            pipe_kwargs["callback_on_step_end"] = callback

        with torch.inference_mode():
            result = self._pipe(**pipe_kwargs)

        return self._save_image(
            result.images[0], output_path,
            seed=seed, steps=steps, guidance=guidance, width=width, height=height,
        )

    # ── 2-phase batch ──────────────────────────────────────────────────────

    def load_for_encode(self):
        self._load_pipeline()
        self._pipe.text_encoder.to("cuda")
        logger.info("Text encoder on GPU — ready for prompt encoding.")

    def encode_prompt(self, prompt: str) -> dict:
        import torch

        with torch.inference_mode():
            prompt_embeds, text_ids = self._pipe.encode_prompt(
                prompt=prompt, device="cuda", num_images_per_prompt=1,
            )
        return {"prompt_embeds": prompt_embeds.cpu(), "text_ids": text_ids.cpu()}

    def start_generation_phase(self):
        import gc
        import torch

        self._pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        self._pipe.vae.to("cuda")
        self._set_execution_device()
        self._optimize_pipe(self._pipe, compile=False)
        logger.info("Generation phase ready (NVFP4 transformer + VAE on GPU).")

    def generate_from_embeddings(
        self, embeddings: dict, *, width: int = 1024, height: int = 1024,
        steps: int = 50, guidance: float = 4.0, seed: int | None = None,
        output_path: Path, callback=None,
    ) -> Path:
        import random
        import torch

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        pipe_kwargs = {
            "prompt_embeds": embeddings["prompt_embeds"].to("cuda"),
            "width": width, "height": height, "num_inference_steps": steps,
            "guidance_scale": guidance, "generator": generator,
        }
        if callback is not None:
            pipe_kwargs["callback_on_step_end"] = callback
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            result = self._pipe(**pipe_kwargs)

        return self._save_image(
            result.images[0], output_path,
            seed=seed, steps=steps, guidance=guidance, width=width, height=height,
        )
