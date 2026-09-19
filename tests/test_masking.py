"""Missing observations contribute exactly zero gradient (PRD §4.5 test_masking).

Guards the never-impute rule: masked entries must not move the loss
value (beyond valid-entry normalization) nor any gradient.
"""

import numpy as np
import pytest
import torch

from bapinnsformer.train.losses import masked_mae, masked_mse


def test_masked_value_ignores_missing():
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.5, 2.0, 99.0, 4.5])
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    # Valid entries: (1-1.5)^2 + 0 + (4-4.5)^2 = 0.5 over 3 valid => 1/6.
    assert float(masked_mse(pred, target, mask)) == pytest.approx(0.5 / 3.0)
    assert float(masked_mae(pred, target, mask)) == pytest.approx((0.5 + 0.5) / 3.0)


def test_masked_entries_have_zero_gradient():
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    masked_mse(pred, target, mask).backward()
    np.testing.assert_allclose(pred.grad.numpy(), [2 / 2, 0.0, 6 / 2, 0.0])


def test_bool_mask_and_all_masked():
    pred = torch.tensor([5.0, 6.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0])
    loss = masked_mse(pred, target, torch.tensor([True, False]))
    assert float(loss) == 25.0
    loss_all = masked_mse(pred, target, torch.tensor([False, False]))
    assert float(loss_all) == 0.0
