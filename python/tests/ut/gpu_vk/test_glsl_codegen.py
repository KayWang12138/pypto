#!/usr/bin/env python3

import torch

from pypto_gpu_vk.compile import build_tensor_graph, compile_to_spirv, lower_to_dispatch, lower_to_shader


def test_glsl_codegen_emits_compute_shader():
    graph = build_tensor_graph("add", torch.randn(16), torch.randn(16))
    dispatch_graph = lower_to_dispatch(graph)
    shader_ir = lower_to_shader(graph)
    _meta, glsl, _spirv = compile_to_spirv(shader_ir, dispatch_graph)
    assert "#version 450" in glsl
    assert "gl_GlobalInvocationID.x" in glsl
