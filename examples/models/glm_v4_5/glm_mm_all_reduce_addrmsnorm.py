#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import multiprocessing as mp
import torch.distributed as dist

# 通信域初始化相关参数
MASTER_IP = "127.0.0.1"
MASTER_PORT = "50001"
WORLD_SIZE = 8
RANK_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
PHYSICAL_TO_LOGICAL = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}

def init_hccl_comm(rank):
    torch_npu.npu.set_device(rank)
    logical_rank = PHYSICAL_TO_LOGICAL[rank]
    dist.init_process_group(
        backend="hccl",
        rank=logical_rank,
        world_size=WORLD_SIZE,
        init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}"
    )
    logical_ranks = [PHYSICAL_TO_LOGICAL[r] for r in RANK_LIST]
    group_handle = dist.new_group(backend="hccl", ranks=logical_ranks)
    group1 = group_handle._get_backend(torch.device("npu")).get_hccl_comm_name(logical_rank)
    groups = []
    groups.append(group1)
    return groups

@allow_in_graph
def mm_all_reduce_addrmsnorm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    input_norm_weight: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    hidden_states_res: torch.Tensor,
    residual_res: torch.Tensor,
    eps: float,
    group_name: str) -> None:
    if isinstance(hidden_states, FakeTensor):
        return
    inputs = {
        hidden_states: [0],
        residual: [0],
        input_norm_weight: [],
        bias: [],
        weight: []
    }
    outputs = {
        hidden_states_res: [0],
        residual_res: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]
    mm_all_reduce_addrmsnorm_kernel(*pto_inputs, *pto_outputs, eps, group_name)
    pypto.runtime._device_synchronize()

@pypto.jit(
    host_options={"only_codegen": True}
)
def mm_all_reduce_addrmsnorm_kernel(x, residual_input, x_gamma, x_bias, weight, hidden_states_out, residual_out, 
                                    eps, group_name):
    calc_dtype = pypto.DT_FP32
    input_dtype = x.dtype
    x_mean_coff = 1.0 / hidden_states_out.shape[-1]
    bs = x.shape[0]
    hidden_size = hidden_states_out.shape[1]
    view_row_shape = 8
    bs_loop = (bs + view_row_shape - 1) // view_row_shape
  
    pypto.set_vec_tile_shapes(hidden_size)
    x_gamma_2d = pypto.reshape(x_gamma, [1,hidden_size], inplace=True)
    x_bias_2d = pypto.reshape(x_bias, [1,hidden_size], inplace=True)

    for bs_idx in pypto.loop(bs_loop, name="LOOP_MM_ALLREDUCE_ADDRMSNORM", idx_name="bs_idx"):
        # 1. CreateShmemTensor
        shmem_data = pypto.tensor()
        shmem_signal = pypto.tensor()
        for _ in pypto.loop(1, name="CREATE_SHMEM_TENSOR", idx_name = "_"):
            pypto.create_shmem_data(group_name, WORLD_SIZE, pypto.DT_FP32, [1, view_row_shape, hidden_size], shmem_data, 0)
            pypto.create_shmem_signal(group_name, shmem_data, shmem_signal)

        for ar_idx in pypto.loop(1, name="LOOP_MM_AR_ARMS_L0", idx_name="ar_idx"):
            tile_in_tensor = pypto.view(x, (view_row_shape, x.shape[1]), [bs_idx * view_row_shape, 0],
                                            valid_shape=[(bs - bs_idx * view_row_shape).min(view_row_shape), x.shape[1]])
            # 2. Matmul
            pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512], True)
            matmul_result = pypto.matmul(tile_in_tensor, weight, x.dtype, b_trans=True)

            # 3. ShmemAllReduce
            pypto.set_vec_tile_shapes(view_row_shape, hidden_size)
            local_rank_id = pypto.get_hccl_rank_id(group_name)
            for dyn_idx in range(WORLD_SIZE):
                shmem_data_tile = pypto.view(shmem_data, [1, 1, view_row_shape, hidden_size], [dyn_idx, 0, 0, 0])
                shmem_signal_tile = pypto.view(shmem_signal, [1, 1, 1, view_row_shape, hidden_size], [dyn_idx, dyn_idx, 0, 0, 0])
                dummy = pypto.shmem_put(matmul_result, shmem_data_tile, matmul_result, pypto.AtomicType.ADD)
                pypto.shmem_signal(dummy, shmem_signal_tile, pypto.AtomicType.ADD)
            local_data_tile = pypto.view(shmem_data, [1, 1, view_row_shape, hidden_size], [local_rank_id, 0, 0, 0])
            local_signal_tile = pypto.view(shmem_signal, [1, 1, 1, view_row_shape, hidden_size], 
                                           [local_rank_id, local_rank_id, 0, 0, 0])
            local_dummy = pypto.wait_until(matmul_result, local_signal_tile, WORLD_SIZE, True)
            pypto.set_vec_tile_shapes(1, hidden_size)
            x_tile = pypto.shmem_get_gm2ub(local_dummy, local_data_tile, input_dtype, pypto.AtomicType.SET)

            # 4. AddRmsNorm
            residual_input_tile = pypto.view(residual_input, (view_row_shape, hidden_states_out.shape[1]), [bs_idx * view_row_shape, 0],
                                            valid_shape=[(bs - bs_idx * view_row_shape).min(view_row_shape),
                                                        hidden_size])
            x_tile_fp32 = pypto.cast(x_tile, calc_dtype)

            # add
            residual_input_tile_fp32 = pypto.cast(residual_input_tile, calc_dtype)
            x_f32 = pypto.add(residual_input_tile_fp32, x_tile_fp32)

            # rms norm
            square = pypto.mul(x_f32, x_f32)
            mean_res = pypto.mul(square, x_mean_coff)
            reduce_asum = pypto.sum(mean_res, -1, True)
            reduce_sum = pypto.add(reduce_asum, eps)
            reduce_sqrt = pypto.sqrt(reduce_sum)
            res_div = pypto.div(x_f32, reduce_sqrt)

            hidden_bf16 = pypto.tensor([view_row_shape, hidden_size], pypto.DT_BF16, "hidden_bf16")
            residual_bf16_tmp = pypto.cast(x_f32, input_dtype)
            for tmp_idx in range(view_row_shape):
                x_gamma_2d_fp32 = pypto.cast(x_gamma_2d, calc_dtype)
                x_bias_2d_fp32 = pypto.cast(x_bias_2d, calc_dtype)
                res_div_single = pypto.view(res_div, [1, hidden_size], [tmp_idx, 0])
                res = pypto.mul(res_div_single, x_gamma_2d_fp32)
                res_add = pypto.add(res, x_bias_2d_fp32)
                x_norm = pypto.cast(res_add, input_dtype)
                hidden_bf16[tmp_idx:tmp_idx + 1, 0:] = x_norm

            residual_out[bs_idx * pypto.symbolic_scalar(view_row_shape):, 0:] = residual_bf16_tmp
            hidden_states_out[bs_idx * pypto.symbolic_scalar(view_row_shape):, 0:] = hidden_bf16

    
def generate_mm_all_reduce_golden_data(m_shape, k_shape, n_shape, data_type):
    in_tensors = []
    weights = []
    out_tensor_golden_fp32 = torch.zeros((m_shape, n_shape), dtype = torch.float32)
    for _ in range(len(RANK_LIST)):
        in_tensor = torch.randn((m_shape, k_shape), dtype=data_type)
        in_tensor.share_memory_()
        in_tensors.append(in_tensor)
        weight = torch.randn((n_shape, k_shape), dtype=data_type)
        weight.share_memory_()
        weights.append(weight)
        matmul_result = torch.matmul(in_tensor, weight.T)
        out_tensor_golden_fp32 += matmul_result.to(torch.float32)
    out_tensor_golden = out_tensor_golden_fp32.to(data_type)
    return in_tensors, weights, out_tensor_golden

def generate_mm_allreduce_rmsnorm_golden(hidden_states, residual, gamma, bias_input, eps):
    x_dtype = residual.dtype
    res_add = residual.to(torch.float32) + hidden_states.to(torch.float32)
    mean_coff = 1.0 / res_add.shape[-1]
    x_f32 = res_add
    square = x_f32 * x_f32
    square = square.sum(dim=-1, keepdim=True)
    mean_res = square * mean_coff
    reduce_sum = mean_res + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt
    res = res_div * gamma.to(torch.float32)
    res = res + bias_input.to(res.dtype)
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
        x_out = x_f32.to(x_dtype)
    return res, x_out

def worker(in_tensor, weight, allreduce_golden, rank):
    torch.manual_seed(42)
    groups = init_hccl_comm(rank)
    batch_size = 8
    h_num = 5120
    eps = 1e-5
    residual_tensor = torch.rand((batch_size, h_num), dtype=torch.bfloat16, device=f'npu:{rank}')
    input_hidden_states = in_tensor.to(f'npu:{rank}')
    input_norm_weight = torch.rand((h_num), dtype=torch.bfloat16, device=f'npu:{rank}')
    bias = torch.rand((h_num), dtype=torch.bfloat16, device=f'npu:{rank}')
    output_hidden_states = torch.empty((batch_size, h_num), dtype=torch.bfloat16, device=f'npu:{rank}')
    output_residual = torch.empty((batch_size, h_num), dtype=torch.bfloat16, device=f'npu:{rank}')

    inputs = {
        input_hidden_states: [0],
        residual_tensor: [0],
        input_norm_weight: [],
        bias: [],
        weight.to(f'npu:{rank}'): []
    }
    outputs = {
        output_hidden_states: [0],
        output_residual: [0]
    }
    pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in inputs.items()]
    pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in outputs.items()]

    mm_all_reduce_addrmsnorm_kernel(*pto_inputs, *pto_outputs, eps, groups[0])
    pypto.runtime._device_synchronize()
    golden_hidden_states, golden_residual = generate_mm_allreduce_rmsnorm_golden(allreduce_golden.to(f'npu:{rank}'), residual_tensor, 
        input_norm_weight, bias, eps)
    assert_allclose(np.array(output_hidden_states.cpu().flatten().tolist()), 
                    np.array(golden_hidden_states.cpu().flatten().tolist()), rtol=8e-3,atol=8e-3)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    processes = []
    in_tensors, weights, out_tensor_golden = generate_mm_all_reduce_golden_data(8, 1536, 5120, torch.bfloat16)
    for idx, rank in enumerate(RANK_LIST):
        p = mp.Process(target=worker, args=(in_tensors[idx], weights[idx], out_tensor_golden, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()