"""Runtime NVFP4 quantizer + pre-quantized weight patcher.

Two helpers that both mutate a transformer's ``nn.Linear`` weights into
``comfy_kitchen.QuantizedTensor`` (TensorCoreNVFP4Layout):

- ``runtime_quantize_transformer_to_nvfp4`` — quantizes bf16 weights on the fly.
  Used when a LoRA has been fused into the bf16 base; the pre-quantized disk
  weights can no longer be applied because they encode the un-fused model.
- ``patch_transformer_nvfp4`` — loads BFL's pre-quantized NVFP4 safetensors,
  remaps keys to the diffusers layout, and installs the QuantizedTensors
  directly. Non-NVFP4 params (norm scales, biases) are copied into place.

Heavy imports (``torch``, ``comfy_kitchen``, ``safetensors``) stay
function-local — nothing at import time.
"""

from __future__ import annotations

from imagecli.engines.nvfp4_state_dict import (
    NVFP4_SUFFIXES,
    convert_nvfp4_state_dict,
)


def runtime_quantize_transformer_to_nvfp4(transformer) -> int:
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
        qt = QuantizedTensor.from_float(w.to("cuda"), layout_cls="TensorCoreNVFP4Layout")
        module.weight = torch.nn.Parameter(qt, requires_grad=False)
        quantized += 1
    return quantized


def patch_transformer_nvfp4(transformer, nvfp4_path: str) -> int:
    """Replace linear layer weights with NVFP4 QuantizedTensors."""
    import torch
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.nvfp4 import TensorCoreNVFP4Layout as NVFP4
    from safetensors import safe_open

    # Load and convert BFL keys to diffusers keys
    with safe_open(nvfp4_path, framework="pt", device="cuda") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}

    converted = convert_nvfp4_state_dict(raw)

    # Group by layer: find layers that have .weight (uint8) + .weight_scale + .weight_scale_2
    layers: dict[str, dict[str, object]] = {}
    for key, tensor in converted.items():
        for suffix in NVFP4_SUFFIXES:
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
        is_nvfp4 = any(key.endswith(s) for s in NVFP4_SUFFIXES)
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
