"""Pseudo-sequence contract tests (PRD §4.5 test_pseudoseq.py).

Under uniform wind, backward characteristics are straight lines;
boundary-hit flags fire exactly when a characteristic exits the
domain; jitter is deterministic given a seed and vanishes for K = 0.
"""

import numpy as np
import pytest

from bapinnsformer.models.pseudoseq import (
    AdvectionBackwardGenerator,
    AdvectionJitterGenerator,
    UniformBackwardGenerator,
    UniformForwardGenerator,
    create_generator,
    get_generator,
)

U, V = 2.0, -1.0
QX, QY, QT = 1000.0, 2000.0, 10.0
BIG = (0.0, 100_000.0, 0.0, 100_000.0)


def _np(tokens):
    import torch

    if isinstance(tokens, torch.Tensor):
        return tokens.detach().cpu().numpy()
    return np.asarray(tokens)


def _tok(tokens):
    """Tokens as (L, 3): scalar queries may return (L, 3) or (1, L, 3)."""
    return _np(tokens).reshape(-1, 3)


def _hit(flags):
    return np.asarray(flags).reshape(-1)


def test_uniform_forward_offsets():
    gen = UniformForwardGenerator(seq_len=4, dt=2.0)
    tok, hit = gen.generate(QX, QY, QT, wind=None)
    tok = _tok(tok)
    assert tok.shape == (4, 3)
    np.testing.assert_allclose(tok[:, 0], QX)
    np.testing.assert_allclose(tok[:, 1], QY)
    np.testing.assert_allclose(tok[:, 2], [QT, QT + 2, QT + 4, QT + 6])
    assert not _hit(hit).any()


def test_uniform_backward_offsets():
    gen = UniformBackwardGenerator(seq_len=3, dt=1.0)
    tok, _ = gen.generate(QX, QY, QT, wind=(U, V))
    tok = _tok(tok)
    np.testing.assert_allclose(tok[:, 2], [QT, QT - 1, QT - 2])
    np.testing.assert_allclose(tok[:, 0], QX)


def test_advection_backward_is_straight_under_uniform_wind():
    gen = AdvectionBackwardGenerator(seq_len=4, dt=1.0, bounds=BIG)
    tok, hit = gen.generate(QX, QY, QT, wind=(U, V))
    tok = _tok(tok)
    for k in range(4):
        assert tok[k, 0] == pytest.approx(QX - U * k, abs=1e-3)
        assert tok[k, 1] == pytest.approx(QY - V * k, abs=1e-3)
        assert tok[k, 2] == pytest.approx(QT - k)
    assert not _hit(hit).any()


def test_query_token_is_first_and_unflagged_inside():
    gen = AdvectionBackwardGenerator(seq_len=3, dt=1.0, bounds=BIG)
    tok, hit = gen.generate(QX, QY, QT, wind=(U, V))
    tok = _tok(tok)
    np.testing.assert_allclose(tok[0], [QX, QY, QT], atol=1e-4)
    assert not bool(_hit(hit)[0])


def test_boundary_flags_and_clamp():
    # Wind blows +x; backward characteristics head −x and exit x < 0.
    bounds = (0.0, 10.0, -100.0, 100.0)
    gen = AdvectionBackwardGenerator(seq_len=5, dt=1.0, bounds=bounds)
    tok, hit = gen.generate(2.0, 0.0, 5.0, wind=(5.0, 0.0))
    tok, hit = _tok(tok), _hit(hit)
    assert hit.shape == (5,)
    assert hit[-1]  # far-past token exited and latched
    assert float(tok[:, 0].min()) >= 0.0  # clamped into domain


def test_advection_needs_wind_for_L_gt_1():
    gen = AdvectionBackwardGenerator(seq_len=3, dt=1.0)
    with pytest.raises(ValueError):
        gen.generate(QX, QY, QT, wind=None)


def test_jitter_zero_K_matches_advection():
    adv = AdvectionBackwardGenerator(seq_len=4, dt=1.0, bounds=BIG)
    jit = AdvectionJitterGenerator(seq_len=4, dt=1.0, bounds=BIG, K=0.0, seed=0)
    ta, _ = adv.generate(QX, QY, QT, wind=(U, V))
    tj, _ = jit.generate(QX, QY, QT, wind=(U, V))
    np.testing.assert_allclose(_np(tj), _np(ta), atol=1e-4)


def test_jitter_seed_reproducible_and_spreads():
    kw = dict(seq_len=6, dt=1.0, K=100.0)
    j1 = AdvectionJitterGenerator(seed=7, **kw)
    j2 = AdvectionJitterGenerator(seed=7, **kw)
    t1, _ = j1.generate(QX, QY, QT, wind=(U, V), bounds=BIG)
    t2, _ = j2.generate(QX, QY, QT, wind=(U, V), bounds=BIG)
    np.testing.assert_array_equal(_np(t1), _np(t2))
    j3 = AdvectionJitterGenerator(seed=8, **kw)
    t3, _ = j3.generate(QX, QY, QT, wind=(U, V), bounds=BIG)
    assert not np.array_equal(_np(t1), _np(t3))


def test_factory_names():
    for name in ("uniform_forward", "uniform_backward", "advection_backward", "advection_jitter"):
        assert create_generator(name, seq_len=2).variant == name
        assert get_generator(name, seq_len=2).variant == name
    with pytest.raises(KeyError):
        create_generator("sideways", seq_len=2)
