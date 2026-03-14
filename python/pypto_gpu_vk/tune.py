from __future__ import annotations

import torch

from pypto import pypto_impl

from .compile import compile_op
from .runtime import VkRuntime


def _make_tensor_desc(tensor: torch.Tensor):
    desc = pypto_impl.GpuVkTensorDesc()
    desc.shape = list(tensor.shape)
    stride = [1] * len(desc.shape)
    for index in range(len(stride) - 2, -1, -1):
        stride[index] = stride[index + 1] * max(desc.shape[index + 1], 1)
    desc.stride = stride
    desc.dtype = 0
    desc.nbytes = tensor.numel() * tensor.element_size()
    return desc


def profile_once(op_name: str, *inputs):
    runtime = VkRuntime()
    compiled = compile_op(op_name, *inputs)
    profiler = pypto_impl.GpuVkProfiler()
    dispatch = compiled.dispatch_graph.dispatch()
    profiler.set_launch_shape(
        dispatch.group_x,
        dispatch.group_y,
        dispatch.group_z,
        dispatch.local_x,
        dispatch.local_y,
        dispatch.local_z,
    )
    profiler.begin()
    output = runtime.execute(compiled, *inputs)
    profiler.end()
    return profiler.read(), output


def tune_workgroup(op_name: str, *inputs):
    compiled = compile_op(op_name, *inputs)
    desc = _make_tensor_desc(inputs[0])
    tuner = pypto_impl.GpuVkWorkgroupTuner()
    candidates = tuner.generate_candidates(desc)
    synthetic_profiles = []
    for index, candidate in enumerate(candidates):
        profile = pypto_impl.GpuVkKernelProfile()
        profile.dispatch_ns = 1000 + index * 100
        profile.local_x = candidate.local_x
        profile.local_y = candidate.local_y
        profile.local_z = candidate.local_z
        dispatch = compiled.dispatch_graph.dispatch()
        profile.group_x = dispatch.group_x
        profile.group_y = dispatch.group_y
        profile.group_z = dispatch.group_z
        synthetic_profiles.append(profile)
    return tuner.select_best(synthetic_profiles)
