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
import multiprocessing as mp
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import torch.distributed as dist

# 通信域初始化相关参数
MASTER_IP = "127.0.0.1"
MASTER_PORT = "50001"
WORLD_SIZE = 8
RANK_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
PHYSICAL_TO_LOGICAL = {0 : 0, 1 : 1, 2 : 2, 3 : 3, 4 : 4, 5 : 5, 6 : 6, 7 : 7}


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
    group = group_handle._get_backend(torch.device("npu")).get_hccl_comm_name(logical_rank)
    return [group]


def matmul_allreduce_addrmsnorm_kernel(bs, ne, h_num, eps, group_name):
    bs = pypto.frontend.dynamic("bs")
    
    @pypto.frontend.jit()
    def kernel(
        x: pypto.Tensor((bs, ne), pypto.DT_BF16),
        matmul_weight: pypto.Tensor((h_num, ne), pypto.DT_BF16),
        residual_input: pypto.Tensor((bs, h_num), pypto.DT_BF16),
        x_gamma: pypto.Tensor((h_num,), pypto.DT_BF16),
        x_bias: pypto.Tensor((h_num,), pypto.DT_BF16),
        hidden_states_out: pypto.Tensor((bs, h_num), pypto.DT_BF16),
        residual_out: pypto.Tensor((bs, h_num), pypto.DT_BF16),
    ):
        calc_dtype = pypto.DT_FP32
        input_dtype = x.dtype
        x_mean_coff = 1.0 / hidden_states_out.shape[-1]
        view_row_shape = 8
        bs_loop = (bs + view_row_shape - 1) // view_row_shape

        pypto.set_vec_tile_shapes(h_num)
        x_gamma_2d = pypto.reshape(x_gamma, [1, h_num], inplace=True)
        x_bias_2d = pypto.reshape(x_bias, [1, h_num], inplace=True)

        for bs_idx in pypto.loop(bs_loop, name="LOOP_MM_ALLREDUCE_ADDRMSNORM", idx_name="bs_idx"):
            # 1. create shmem tesnor
            shmem_shape = [1, view_row_shape, h_num]
            shmem_data, shmem_signal = pypto.distributed.create_shmem_tensor(
                group_name, WORLD_SIZE, shmem_shape, pypto.DT_FP32)
            shmem_barrier_signal = pypto.distributed.create_shmem_barrier_signal(group_name, WORLD_SIZE)
            
            for _ in pypto.loop(1, name="LOOP_MM_AR_ARMS_L0", idx_name="_"):
                tile_in_tensor = pypto.view(x, (view_row_shape, x.shape[1]), [bs_idx * view_row_shape, 0],
                    valid_shape=[(bs - bs_idx * view_row_shape).min(view_row_shape), x.shape[1]])
                
                # 2. clear data
                pypto.set_vec_tile_shapes(view_row_shape, h_num)
                data_clear_dummy = pypto.distributed.shmem_clear(
                    shmem_data, shmem_shape, [0, 0, 0], False, [tile_in_tensor])
                signal_clear_dummy = pypto.distributed.shmem_clear(
                    shmem_signal, shmem_shape, [0, 0, 0], True, [tile_in_tensor])
                pypto.set_vec_tile_shapes(1, 8)
                barrier_dummy = pypto.distributed.shmem_barrier_all(
                    shmem_barrier_signal, group_name, [data_clear_dummy, signal_clear_dummy])

                # 3. matmul
                pypto.set_cube_tile_shapes([8, 8], [128, 256], [256, 512], True)
                matmul_result = pypto.matmul(tile_in_tensor, matmul_weight, x.dtype, b_trans=True)

                # 4. allreduce
                pypto.set_vec_tile_shapes(view_row_shape, h_num)
                for dyn_idx in range(WORLD_SIZE):
                    put_dummy = pypto.distributed.shmem_put(matmul_result, [0, 0, 0], shmem_data, dyn_idx,
                        pypto.AtomicType.ADD, [barrier_dummy])
                    pypto.distributed.shmem_signal(shmem_signal, dyn_idx, [shmem_shape], [[0, 0, 0]],
                        pypto.AtomicType.ADD, [put_dummy])
                wait_dummy = pypto.distributed.shmem_wait(
                    shmem_signal, [shmem_shape], [[0, 0, 0]], WORLD_SIZE, True, [tile_in_tensor])
                my_pe = pypto.distributed.my_symbolic_pe(group_name)
                pypto.set_vec_tile_shapes(1, h_num)
                reduce_out = pypto.experimental.shmem_load(shmem_data, my_pe, shmem_shape, [0, 0, 0], [wait_dummy])
                x_tile = pypto.cast(reduce_out, pypto.DT_BF16)

                # 5. AddRmsNorm
                residual_input_tile = pypto.view(residual_input, (view_row_shape, h_num), [bs_idx * view_row_shape, 0],
                                                valid_shape=[(bs - bs_idx * view_row_shape).min(view_row_shape), h_num])
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

                hidden_bf16 = pypto.tensor([view_row_shape, h_num], pypto.DT_BF16, "hidden_bf16")
                residual_bf16_tmp = pypto.cast(x_f32, input_dtype)
                for tmp_idx in range(view_row_shape):
                    x_gamma_2d_fp32 = pypto.cast(x_gamma_2d, calc_dtype)
                    x_bias_2d_fp32 = pypto.cast(x_bias_2d, calc_dtype)
                    res_div_single = pypto.view(res_div, [1, h_num], [tmp_idx, 0])
                    res = pypto.mul(res_div_single, x_gamma_2d_fp32)
                    res_add = pypto.add(res, x_bias_2d_fp32)
                    x_norm = pypto.cast(res_add, input_dtype)
                    hidden_bf16[tmp_idx:tmp_idx + 1, 0:] = x_norm

                residual_out[bs_idx * pypto.symbolic_scalar(view_row_shape):, 0:] = residual_bf16_tmp
                hidden_states_out[bs_idx * pypto.symbolic_scalar(view_row_shape):, 0:] = hidden_bf16
    return kernel


def generate_golden_data():
    # 设置参数
    bs = 8
    ne = 1536
    h_num = 5120
    torch.manual_seed(42)

    #构造每张卡上需要的数据
    input_datas = []
    for _ in range(len(RANK_LIST)):
        in_tensor = torch.randn((bs, ne), dtype=torch.bfloat16).share_memory_()
        matmul_weight = torch.randn((h_num, ne), dtype=torch.bfloat16).share_memory_()
        residual_input = torch.randn((bs, h_num), dtype=torch.bfloat16).share_memory_()
        gamma = torch.randn((h_num), dtype=torch.bfloat16).share_memory_()
        bias = torch.randn((h_num), dtype=torch.bfloat16).share_memory_()
        eps = 1e-5
        input_data = [in_tensor, matmul_weight, residual_input, gamma, bias, eps]
        input_datas.append(input_data)
    output_datas = matmul_allreduce_addrmsnorm_result_golden(bs, h_num, input_datas)
    return input_datas, output_datas


def matmul_allreduce_addrmsnorm_result_golden(bs, num, input_datas):
    output_datas = []
    # 计算 matmul & allreduce 结果， 该结果所有卡上一致
    matmul_allreduce_result_fp32 = torch.zeros((bs, num), dtype=torch.float32)
    for input_data in input_datas:
        in_tensor, matmul_weight = input_data[:2]
        matmul_result = torch.matmul(in_tensor, matmul_weight.T)
        matmul_allreduce_result_fp32 += matmul_result.to(torch.float32)

    # 计算各卡上addrmsnorm之后的结果
    for input_data in input_datas:
        residual_input, gamma, bias, eps = input_data[-4:]
        res_add = residual_input.to(torch.float32) + matmul_allreduce_result_fp32
        mean_coff = 1.0 / res_add.shape[-1]
        x_f32 = res_add
        square = x_f32 * x_f32
        square = square.sum(dim=-1, keepdim=True)
        mean_res = square * mean_coff
        reduce_sum = mean_res + eps
        reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = x_f32 / reduce_sqrt
        res = res_div * gamma.to(torch.float32)
        res = res + bias.to(res.dtype)
        output_data = [res.to(torch.bfloat16), x_f32.to(torch.bfloat16)]
        output_datas.append(output_data)
    return output_datas


def test_matmul_allreduce_addrmsnorm(intput_data, output_data, rank):
    groups = init_hccl_comm(rank)
    device = f'npu:{rank}'
    in_tensor, matmul_weight, residual_input, gamma, bias, eps = intput_data
    golden_hidden_states, golden_residual = output_data

    output_hidden_states = torch.empty(residual_input.shape, dtype=torch.bfloat16, device=device)
    output_residual = torch.empty(residual_input.shape, dtype=torch.bfloat16, device=device)

    inputs = [in_tensor.to(device), matmul_weight.to(device), residual_input.to(device), gamma.to(device), 
        bias.to(device), output_hidden_states, output_residual]

    bs, ne = in_tensor.shape
    h_num = output_hidden_states.shape[1]

    matmul_allreduce_addrmsnorm_kernel(bs, ne, h_num, eps, groups[0])(*inputs)

    assert_allclose(np.array(output_hidden_states.cpu().flatten().tolist()), 
                    np.array(golden_hidden_states.cpu().flatten().tolist()), rtol=8e-3, atol=8e-3)


@allow_in_graph
def matmul_allreduce_addrmsnorm(
    hidden_states: torch.Tensor,
    matmul_weight: torch.Tensor,
    residual: torch.Tensor,
    input_norm_weight: torch.Tensor,
    bias: torch.Tensor,
    hidden_states_res: torch.Tensor,
    residual_res: torch.Tensor,
    eps: float,
    group_name: str) -> None:
    if isinstance(hidden_states, FakeTensor):
        return
    
    inputs = [hidden_states, matmul_weight, residual, input_norm_weight, bias, hidden_states_res, residual_res]

    bs, ne = hidden_states.shape
    h_num = hidden_states_res.shape[1]

    matmul_allreduce_addrmsnorm_kernel(bs, ne, h_num, eps, group_name)(*inputs)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    processes = []
    input_datas, output_datas = generate_golden_data()
    for idx, rank in enumerate(RANK_LIST):
        p = mp.Process(target=test_matmul_allreduce_addrmsnorm, args=(input_datas[idx], output_datas[idx], rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()