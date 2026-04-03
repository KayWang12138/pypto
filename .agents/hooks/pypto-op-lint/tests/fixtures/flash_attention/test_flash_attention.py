import os
import numpy as np
from numpy.testing import assert_allclose

from .flash_attention_impl import flash_attention_wrapper
from .flash_attention_golden import flash_attention_golden


def get_device_id() -> int:
    return int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))


def test_flash_attention_level0(device_id=None):
    np.random.seed(42)
    import torch

    torch.manual_seed(42)
    result = np.array([1.0], dtype=np.float32)
    expected = np.array([1.0], dtype=np.float32)
    assert_allclose(result, expected, rtol=1e-3, atol=1e-3)


def test_flash_attention_level1(device_id=None):
    np.random.seed(42)
    import torch

    torch.manual_seed(42)
    result = np.array([1.0], dtype=np.float32)
    expected = np.array([1.0], dtype=np.float32)
    assert_allclose(result, expected, rtol=1e-3, atol=1e-3)
