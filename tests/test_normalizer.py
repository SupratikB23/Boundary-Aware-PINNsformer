"""Round-trip scaling exactness (PRD §4.5 test_normalizer.py)."""

import math

import numpy as np
import pytest
import torch

from bapinnsformer.models.normalizer import Normalizer


@pytest.fixture()
def norm() -> Normalizer:
    return Normalizer(
        x_range=(0.0, 60_000.0),
        y_range=(-30_000.0, 30_000.0),
        t_range=(0.0, 24.0 * 3600.0),
        c_scale=200.0,
        c_shift=10.0,
    )


def test_edges_map_to_pm_one(norm: Normalizer):
    assert norm.normalize_x(0.0) == pytest.approx(-1.0)
    assert norm.normalize_x(60_000.0) == pytest.approx(1.0)
    assert norm.normalize_y(0.0) == pytest.approx(0.0)
    assert norm.normalize_t(0.0) == pytest.approx(-1.0)


def test_roundtrip_numpy(norm: Normalizer):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 60_000, 50)
    y = rng.uniform(-30_000, 30_000, 50)
    t = rng.uniform(0, 24 * 3600, 50)
    for fwd, inv, v in (
        (norm.normalize_x, norm.denormalize_x, x),
        (norm.normalize_y, norm.denormalize_y, y),
        (norm.normalize_t, norm.denormalize_t, t),
        (norm.normalize_c, norm.denormalize_c, rng.uniform(0, 500, 50)),
    ):
        back = inv(fwd(v))
        np.testing.assert_allclose(np.asarray(back), v, rtol=1e-12, atol=1e-9)


def test_roundtrip_torch_preserves_type(norm: Normalizer):
    x = torch.linspace(0, 60_000, 11, dtype=torch.float64)
    xh = norm.normalize_x(x)
    assert isinstance(xh, torch.Tensor)
    assert torch.all(xh <= 1.0) and torch.all(xh >= -1.0)
    torch.testing.assert_close(norm.denormalize_x(xh).double(), x.double())


def test_state_dict_roundtrip(norm: Normalizer):
    clone = Normalizer.from_state_dict(norm.state_dict())
    assert clone.state_dict() == norm.state_dict()
    assert math.isclose(clone.normalize_c(310.0), norm.normalize_c(310.0))


def test_bad_bounds_rejected():
    with pytest.raises(ValueError):
        Normalizer(x_range=(5.0, 5.0), y_range=(0.0, 1.0), t_range=(0.0, 1.0))
