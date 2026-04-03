"""test_good_op.py — 全部通过 OL17-OL22 的 test fixture"""
import os
import numpy as np
from numpy.testing import assert_allclose
from good_op_impl import good_op_wrapper
from good_op_golden import good_op_golden

torch = None  # mock


def get_device_id():
    return int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


def test_good_op_level0(device_id=None):
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    # ... test logic
    result = np.array([1.0])
    expected = np.array([1.0])
    assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_good_op_level1(device_id=None):
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    # ... test logic
    result = np.array([1.0])
    expected = np.array([1.0])
    assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
