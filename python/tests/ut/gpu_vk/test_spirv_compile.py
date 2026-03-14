#!/usr/bin/env python3

import torch

from pypto_gpu_vk.compile import compile_op


def test_compile_op_produces_pseudo_spirv():
    compiled = compile_op("mul", torch.randn(32), torch.randn(32))
    assert compiled.artifact.spirv[0] == 0x07230203
    assert len(compiled.dump_spirv()) >= 20
