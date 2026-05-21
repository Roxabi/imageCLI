"""Free helpers shared across engine implementations."""

from __future__ import annotations


def set_execution_device(pipe) -> None:
    """Patch pipeline execution device to cuda (idempotent class-level monkey-patch).

    Pipeline.device / _execution_device derives from the first registered module
    (text_encoder, on CPU). We patch the class-level property to honor an
    instance-level override, so single-image and 2-phase batch paths report cuda
    even when the encoder has been offloaded.
    """
    import torch

    pipe._execution_device_override = torch.device("cuda")  # type: ignore[attr-defined]
    orig_cls = type(pipe)
    if not hasattr(orig_cls, "_orig_execution_device"):
        orig_cls._orig_execution_device = orig_cls._execution_device  # type: ignore[attr-defined]
        orig_cls._execution_device = property(  # type: ignore[attr-defined]
            lambda p: (
                getattr(p, "_execution_device_override", None)
                or orig_cls._orig_execution_device.fget(p)  # type: ignore[attr-defined]
            )
        )
