"""CPU-only guard for the PuLID Klein v2 Strategy B invariant.

The dim-projection pair (``proj_up`` / ``proj_down``) MUST remain ephemeral —
constructed inside ``patch_flux2`` at inference time, never attached to
``PuLIDFlux2`` as sub-modules. If a future refactor promotes them to model
attributes, they would be included in ``load_state_dict`` targets and either
miss the trained CA weights entirely or clobber the intentional orthogonal
initialization. See ``_pulid/klein_patching.py`` top-of-file docstring for
the full rationale.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from imagecli.engines._pulid.klein_modules import PuLIDFlux2  # noqa: E402
from imagecli.engines._pulid.klein_patching import _make_projections  # noqa: E402


def test_pulid_flux2_has_no_projection_attrs() -> None:
    """PuLIDFlux2 must not carry proj_up/proj_down as sub-modules."""
    model = PuLIDFlux2(dim=64, n_double_ca=2, n_single_ca=2)

    assert hasattr(model, "id_former")
    assert hasattr(model, "double_ca")
    assert hasattr(model, "single_ca")

    assert not hasattr(model, "proj_up"), (
        "proj_up leaked onto PuLIDFlux2 — Strategy B requires projections to be ephemeral"
    )
    assert not hasattr(model, "proj_down"), (
        "proj_down leaked onto PuLIDFlux2 — Strategy B requires projections to be ephemeral"
    )

    state_keys = set(model.state_dict().keys())
    projection_keys = {k for k in state_keys if "proj_up" in k or "proj_down" in k}
    assert not projection_keys, f"projection weights in state_dict: {projection_keys}"


def test_make_projections_returns_none_when_dims_match() -> None:
    """No projection pair when pulid_dim == model_dim."""
    result = _make_projections(
        pulid_dim=3072,
        model_dim=3072,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert result is None


def test_make_projections_returns_orthogonal_pair_when_dims_mismatch() -> None:
    """Klein 4B (3072) ↔ PuLID weights (4096) produces an orthogonal projection pair."""
    result = _make_projections(
        pulid_dim=4096,
        model_dim=3072,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert result is not None
    proj_up, proj_down = result

    assert proj_up.in_features == 3072
    assert proj_up.out_features == 4096
    assert proj_up.bias is None

    assert proj_down.in_features == 4096
    assert proj_down.out_features == 3072
    assert proj_down.bias is None

    # Orthogonal init: W @ W.T ≈ I for the smaller dimension.
    w_up = proj_up.weight.detach()
    identity_up = w_up.T @ w_up
    assert torch.allclose(identity_up, torch.eye(3072), atol=1e-5), (
        "proj_up should be orthogonally initialized"
    )

    w_down = proj_down.weight.detach()
    identity_down = w_down @ w_down.T
    assert torch.allclose(identity_down, torch.eye(3072), atol=1e-5), (
        "proj_down should be orthogonally initialized"
    )
