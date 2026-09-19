"""FD solver: mass conservation, CFL guard, inflow response (PRD §4.5)."""

import numpy as np
import pytest

from bapinnsformer.data.synthetic import SyntheticSolver, check_cfl


def _solver(**kw):
    args = dict(nx=12, ny=12, dx=1000.0, dy=1000.0, dt=10.0, K=0.0, lam=0.0)
    args.update(kw)
    return SyntheticSolver(**args)


def test_static_field_conserves_mass_exactly():
    # Zero-gradient-compatible edges: the BC projection is then identity,
    # so zero dynamics (u=v=K=lam=S=0) must preserve C bit-for-bit.
    s = _solver()
    rng = np.random.default_rng(0)
    C = rng.uniform(10, 100, (12, 12))
    C[0, :] = C[1, :]
    C[-1, :] = C[-2, :]
    C[:, 0] = C[:, 1]
    C[:, -1] = C[:, -2]
    m0 = s.total_mass(C)
    Cn = s.run(C, 0.0, 0.0, S=0.0, n_steps=5)
    np.testing.assert_allclose(Cn, C, rtol=1e-12, atol=1e-12)
    assert s.total_mass(Cn) == pytest.approx(m0, rel=1e-12)


def test_uniform_field_preserved_under_wind():
    s = _solver()
    C = np.full((12, 12), 42.0)
    Cn = s.run(C, 3.0, -1.0, S=0.0, n_steps=4)
    np.testing.assert_allclose(Cn, C, rtol=1e-12, atol=1e-12)


def test_outflow_removes_mass_and_stays_nonnegative():
    s = _solver()
    C = np.full((12, 12), 50.0)
    Cn = s.run(C, 5.0, 0.0, S=0.0, n_steps=20)
    assert float(Cn.min()) >= 0.0
    assert s.total_mass(Cn) <= s.total_mass(C) + 1e-9


def test_prescribed_west_inflow_adds_mass():
    s = _solver()
    C = np.zeros((12, 12))
    Cn = s.run(C, 5.0, 0.0, S=0.0, Cb={"west": 100.0}, n_steps=10)
    assert s.total_mass(Cn) > 0.0


def test_cfl_guard_raises():
    with pytest.raises(ValueError, match="CFL"):
        check_cfl(50.0, 0.0, 100.0, 1000.0, 1000.0, 1e6, cfl=0.9)
    # CFL-safe call returns limits without raising.
    lim = check_cfl(1.0, 0.0, 10.0, 1000.0, 1000.0, 1.0, cfl=0.9)
    assert lim["stable"] > 0


def test_step_rejects_bad_shape():
    s = _solver()
    with pytest.raises(ValueError):
        s.step(np.zeros((4, 4)), 0.0, 0.0)
