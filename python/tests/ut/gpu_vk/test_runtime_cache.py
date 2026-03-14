#!/usr/bin/env python3

import torch

from pypto_gpu_vk.compile import compile_op
from pypto_gpu_vk.runtime import VkRuntime


def test_runtime_runner_cache_tracks_pipeline_reuse():
    runtime = VkRuntime(enable_cache=True)
    compiled = compile_op("add", torch.ones(64), torch.ones(64))
    output0 = runtime.execute(compiled, torch.ones(64), torch.ones(64))
    output1 = runtime.execute(compiled, torch.ones(64), torch.ones(64))
    assert runtime.pipeline_cache_entry_count() >= 1
    assert runtime.pipeline_cache_hit_count() >= 1
    assert torch.allclose(output0, output1)
