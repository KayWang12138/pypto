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
    assert _spirv[0] == 0x07230203


def test_glsl_codegen_emits_matmul_shader():
    graph = build_tensor_graph("matmul", torch.randn(8, 12), torch.randn(12, 6))
    dispatch_graph = lower_to_dispatch(graph)
    shader_ir = lower_to_shader(graph)
    meta, glsl, _spirv = compile_to_spirv(shader_ir, dispatch_graph)
    assert meta.dispatch.group_y >= 1
    assert [item.name for item in meta.push_constants] == ["B", "M", "N", "K", "lhsBatchStride", "rhsBatchStride"]
    assert "shared float Asub[16][16];" in glsl
    assert "shared float Bsub[16][16];" in glsl
    assert "gl_LocalInvocationID.x" in glsl
    assert "gl_WorkGroupID.y" in glsl
    assert "uint batch = gl_WorkGroupID.z;" in glsl
    assert "barrier();" in glsl
    assert "for (uint tile = 0; tile < tileCount; ++tile)" in glsl
    assert _spirv[0] == 0x07230203


def test_glsl_codegen_emits_batched_matmul_dispatch():
    graph = build_tensor_graph("matmul", torch.randn(4, 8, 12), torch.randn(4, 12, 6))
    dispatch_graph = lower_to_dispatch(graph)
    shader_ir = lower_to_shader(graph)
    meta, glsl, _spirv = compile_to_spirv(shader_ir, dispatch_graph)
    assert meta.dispatch.group_z == 4
    assert "uint aBase = batch * pc.lhsBatchStride;" in glsl
    assert "uint bBase = batch * pc.rhsBatchStride;" in glsl
    assert "uint cBase = batch * pc.M * pc.N;" in glsl
    assert _spirv[0] == 0x07230203


def test_glsl_codegen_emits_batched_vector_matmul_dispatch():
    graph = build_tensor_graph("matmul", torch.randn(12), torch.randn(4, 12, 6))
    dispatch_graph = lower_to_dispatch(graph)
    shader_ir = lower_to_shader(graph)
    meta, glsl, _spirv = compile_to_spirv(shader_ir, dispatch_graph)
    assert meta.dispatch.group_y == 1
    assert meta.dispatch.group_z == 4
    assert "uint row = gl_WorkGroupID.y * 16u + localRow;" in glsl
    assert "if (row < pc.M && col < pc.N)" in glsl
    assert _spirv[0] == 0x07230203
