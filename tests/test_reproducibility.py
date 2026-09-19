"""Same seed + config ⇒ same loss trajectory (PRD §4.5 test_reproducibility.py)."""

import random

import numpy as np
import torch

from bapinnsformer.utils.seeding import seed_all


def _draws():
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.randn(()).item()),
        torch.randn(8).tolist(),
    )


def _tiny_trajectory(seed: int, steps: int = 20):
    seed_all(seed)
    w = torch.randn(4, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.01)
    traj = []
    x = torch.linspace(-1, 1, 4)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((w * x - torch.sin(x)) ** 2).mean()
        loss.backward()
        opt.step()
        traj.append(float(loss.item()))
    return traj


def test_seed_all_reproduces_draws():
    seed_all(123)
    a = _draws()
    seed_all(123)
    b = _draws()
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
    assert np.allclose(a[3], b[3])


def test_different_seeds_differ():
    seed_all(1)
    a = _draws()
    seed_all(2)
    b = _draws()
    assert a != b


def test_loss_trajectory_stable_to_tolerance():
    t1 = _tiny_trajectory(0)
    t2 = _tiny_trajectory(0)
    assert len(t1) == len(t2) == 20
    np.testing.assert_allclose(t1, t2, rtol=0, atol=0)
