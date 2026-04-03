"""test_partial_bad_op.py — 合规的 test"""
import os
import torch
import numpy as np
from numpy.testing import assert_allclose
from partial_bad_op_impl import partial_bad_op_wrapper
from partial_bad_op_golden import partial_bad_op_golden


def get_device_id():
    return int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


def test_partial_bad_op_level0(device_id=None):
    torch.manual_seed(42)
    a = torch.randn(8, dtype=torch.float32)
    result = partial_bad_op_wrapper(a)
    expected = partial_bad_op_golden(a)
    assert_allclose(result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)


def test_partial_bad_op_level1(device_id=None):
    torch.manual_seed(42)
    a = torch.randn(1024, dtype=torch.float32)
    result = partial_bad_op_wrapper(a)
    expected = partial_bad_op_golden(a)
    assert_allclose(result.numpy(), expected.numpy(), rtol=1e-3, atol=1e-3)
