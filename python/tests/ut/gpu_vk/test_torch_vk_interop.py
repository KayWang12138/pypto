#!/usr/bin/env python3

import torch

from pypto_gpu_vk.converter import to_torch, torch_to_vk_cpu_staging


def test_torch_vk_cpu_staging_round_trip_shape():
    tensor = torch.randn(4, 8)
    storage = torch_to_vk_cpu_staging(tensor)
    restored = to_torch(storage)
    assert storage.mode_name == "cpu_staging"
    assert tuple(restored.shape) == tuple(tensor.shape)
