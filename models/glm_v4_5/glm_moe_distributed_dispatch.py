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
GLM-4.5 Distributed MoE Dispatch Module

This module implements the distributed token dispatch stage for
MoE in an expert parallel setting, based on open shared memory.

Main Functions:
    - moe_distributed_dispatch_kernel: JIT compiled dispatch kernel
"""
import dataclasses
from typing import Callable

import multiprocessing as mp
import numpy as np
import torch
import torch.distributed as dist
import torch_npu

import pypto

np.random.seed(0)
torch.manual_seed(0)

TensorList = list[torch.Tensor, ...]

MASTER_IP = '127.0.0.1'
MASTER_PORT = "50001"
WORLD_SIZE = 4
PHYSICAL_START_DEVICE_ID = 0
LOGICAL_RANKS = list(range(WORLD_SIZE))


def check_cond(cond, msg):
    if not cond:
        raise ValueError(msg)


def init_hccl_comm(logical_rank: int) -> list[str, ...]:
    physical_device_id = PHYSICAL_START_DEVICE_ID + logical_rank
    torch_npu.npu.set_device(physical_device_id)
    dist.init_process_group(
        backend='hccl',
        rank=logical_rank,
        world_size=WORLD_SIZE,
        init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}',
    )
    group_handle = dist.new_group(backend='hccl', ranks=LOGICAL_RANKS)
    group_name = group_handle._get_backend(torch.device('npu')).get_hccl_comm_name(logical_rank)
    return [group_name]


@dataclasses.dataclass(frozen=True)
class MoeCase:
    batch_size: int
    hidden_size: int
    moe_expert_num: int
    topk: int
    pypto_data_type: pypto.DataType
    torch_data_type: torch.dtype
    world_size: int

    def __post_init__(self):
        check_cond(
            self.topk <= self.moe_expert_num,
            f'topk ({self.topk}) must be <= moe_expert_num ({self.moe_expert_num})'
        )


@dataclasses.dataclass(frozen=True)
class MoeDispatchOperand:
    x: torch.Tensor
    expert_ids: torch.Tensor
    expand_x: torch.Tensor
    assist_info_for_combine: torch.Tensor
    expert_token_nums: torch.Tensor
    recv_counts: torch.Tensor
    rank_id: int


@dataclasses.dataclass(frozen=True)
class MoeDispatchGolden:
    x_list: TensorList
    moe_expert_ids_list: TensorList
    expand_x_list: TensorList
    assist_info_for_combine_list: TensorList
    recv_counts_list: TensorList
    expert_token_nums_list: TensorList


def generate_random_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    float_dtypes = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    int_dtypes = (torch.int8, torch.int16, torch.int32, torch.int64)
    if dtype in float_dtypes:
        return torch.randn(shape, dtype=dtype)
    elif dtype in int_dtypes:
        return torch.randint(-10, 10, size=shape, dtype=dtype)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}. Supported: {float_dtypes + int_dtypes}')


def generate_inputs(moe_case: MoeCase) -> tuple[TensorList, TensorList, TensorList]:
    x_list = [
        generate_random_tensor((moe_case.batch_size, moe_case.hidden_size), moe_case.torch_data_type)
        for _ in range(moe_case.world_size)
    ]
    moe_expert_ids_list = []
    topk_expert_scales_list = []

    for _ in range(moe_case.world_size):
        expert_scores = generate_random_tensor((moe_case.batch_size, moe_case.moe_expert_num), torch.float32)
        topk_expert_scores, moe_expert_ids = expert_scores.topk(k=moe_case.topk)
        topk_expert_scales = topk_expert_scores.softmax(dim=-1)
        topk_expert_scales_list.append(topk_expert_scales)
        moe_expert_ids = moe_expert_ids.to(dtype=torch.int32)
        moe_expert_ids_list.append(moe_expert_ids)

    return x_list, moe_expert_ids_list, topk_expert_scales_list


def get_moe_expert_num_per_rank(moe_case: MoeCase) -> int:
    return (moe_case.moe_expert_num + moe_case.world_size - 1) // moe_case.world_size


def get_moe_expert_rank_id_and_expert_offset(expert_id: int, moe_expert_num_per_rank: int) -> tuple[int, int]:
    return divmod(expert_id, moe_expert_num_per_rank)


def get_dispatch_output_row(moe_case: MoeCase) -> int:
    max_send_token_num = moe_case.topk * moe_case.batch_size * moe_case.world_size
    max_receive_token_num = moe_case.batch_size * moe_case.moe_expert_num
    return min(max_send_token_num, max_receive_token_num)


def dispatch_tokens(
    moe_case: MoeCase,
    x_list: TensorList,
    moe_expert_ids_list: TensorList,
    generate_for_dispatch: bool,
) -> tuple[TensorList, TensorList, TensorList, TensorList]:
    moe_expert_num_per_rank = get_moe_expert_num_per_rank(moe_case)
    expand_x_per_expert = [[[] for _ in range(moe_expert_num_per_rank)] for _ in range(moe_case.world_size)]
    assist_info_for_combine_per_expert = [
        [[] for _ in range(moe_expert_num_per_rank)]
        for _ in range(moe_case.world_size)
    ]

    for sending_rank_id, (x, moe_expert_ids) in enumerate(zip(x_list, moe_expert_ids_list)):
        for token_id, (token, topk_moe_expert_ids) in enumerate(zip(x, moe_expert_ids)):
            for k_offset, moe_expert_id in enumerate(topk_moe_expert_ids):
                receiving_expert_id = moe_expert_id.item()
                receiving_rank_id, expert_offset = get_moe_expert_rank_id_and_expert_offset(
                    receiving_expert_id, moe_expert_num_per_rank)
                expand_x_per_expert[receiving_rank_id][expert_offset].append(token)
                assist_info_for_combine_per_expert[receiving_rank_id][expert_offset].append(
                    (sending_rank_id, token_id, k_offset))

    row = get_dispatch_output_row(moe_case)
    combine_info_col = 64 if generate_for_dispatch else 3
    expand_x_per_rank = []
    assist_info_for_combine_per_rank = []
    expert_token_nums_per_rank = []
    recv_counts_per_rank = []
    for rank_id in range(moe_case.world_size):
        fixed_shape_expand_x = torch.zeros((row, moe_case.hidden_size), dtype=moe_case.torch_data_type)
        fixed_shape_assist_info_for_combine = torch.zeros((row, combine_info_col), dtype=torch.int32)
        expert_token_nums = torch.zeros([moe_expert_num_per_rank], dtype=torch.int32)
        offset = 0

        for expert_offset in range(moe_expert_num_per_rank):
            tokens = expand_x_per_expert[rank_id][expert_offset]
            if tokens:
                actual_expand_x = torch.stack(tokens, dim=0)
                end = offset + actual_expand_x.size(0)
                fixed_shape_expand_x[offset:end] = actual_expand_x
                actual_assist_info_for_combine = torch.tensor(
                    assist_info_for_combine_per_expert[rank_id][expert_offset],
                    dtype=torch.int32,
                )
                fixed_shape_assist_info_for_combine[offset:end, -3:] = actual_assist_info_for_combine
                expert_token_nums[expert_offset] = actual_expand_x.size(0)
                offset = end

        expand_x_per_rank.append(fixed_shape_expand_x)
        assist_info_for_combine_per_rank.append(fixed_shape_assist_info_for_combine)
        expert_token_nums_per_rank.append(expert_token_nums)
        recv_counts_per_rank.append(expert_token_nums.sum().unsqueeze(0))
    return expand_x_per_rank, assist_info_for_combine_per_rank, expert_token_nums_per_rank, recv_counts_per_rank


def combine_tokens(
    moe_case: MoeCase,
    expand_x_list: TensorList,
    assist_info_for_combine_list: TensorList,
    recv_counts_list: TensorList,
    expert_scales_list: TensorList,
) -> TensorList:
    out_golden_list = []
    moe_expert_tokens_list = [
        torch.zeros([moe_case.batch_size, moe_case.topk, moe_case.hidden_size], dtype=moe_case.torch_data_type)
        for _ in range(moe_case.world_size)
    ]

    for expand_x, assist_info_for_combine, recv_counts in zip(
        expand_x_list, assist_info_for_combine_list, recv_counts_list
    ):
        for row_index in range(recv_counts.item()):
            token = expand_x[row_index]
            dispatch_sending_rank_id, token_id, k_offset = assist_info_for_combine[row_index]
            moe_expert_tokens_list[dispatch_sending_rank_id][token_id, k_offset] = token

    for moe_expert_tokens, expert_scales in zip(moe_expert_tokens_list, expert_scales_list):
        out_golden = (
            expert_scales.unsqueeze(1).
            matmul(moe_expert_tokens.to(torch.float32))
            .squeeze(1)
            .to(torch.bfloat16)
        )
        out_golden_list.append(out_golden)

    return out_golden_list


def generate_dispatch_golden(moe_case: MoeCase) -> MoeDispatchGolden:
    x_list, moe_expert_ids_list, _ = generate_inputs(moe_case)
    expand_x_list, assist_info_for_combine_list, expert_token_nums_list, recv_counts_list = dispatch_tokens(
        moe_case, x_list, moe_expert_ids_list, True)
    return MoeDispatchGolden(
        x_list,
        moe_expert_ids_list,
        expand_x_list,
        assist_info_for_combine_list,
        recv_counts_list,
        expert_token_nums_list,
    )


def generate_combine_golden(
    moe_case: MoeCase
) -> tuple[TensorList, TensorList, TensorList, TensorList, TensorList]:
    x_list, moe_expert_ids_list, expert_scales_list = generate_inputs(moe_case)
    (
        expand_x_list,
        assist_info_for_combine_list,
        expert_token_nums_list,
        recv_counts_list,
    ) = dispatch_tokens(moe_case, x_list, moe_expert_ids_list, False)
    out_golden_list = combine_tokens(
        moe_case,
        expand_x_list,
        assist_info_for_combine_list,
        recv_counts_list,
        expert_scales_list,
    )
    return expand_x_list, assist_info_for_combine_list, recv_counts_list, expert_scales_list, out_golden_list


def align_up(value: int, alignment: int) -> int:
    check_cond(
        alignment > 0 and (alignment & (alignment - 1)) == 0,
        f'alignment must be a power of two, but got {alignment}',
    )
    return (value + alignment - 1) & ~(alignment - 1)


def dispatch_calc_occurrences(
    expert_ids: pypto.Tensor, expert_id: pypto.SymbolicScalar, calc_index: int,
) -> pypto.Tensor:
    expert_ids_dup = pypto.full(expert_ids.shape, expert_id, pypto.DT_INT32)
    sub_result = pypto.sub(expert_ids_dup, expert_ids)
    sub_result_fp32 = pypto.cast(sub_result, pypto.DT_FP32, pypto.CastMode.CAST_TRUNC)
    abs_sub_result = pypto.abs(sub_result_fp32)
    abs_sub_result_int32 = pypto.cast(abs_sub_result, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
    count_of_equals = pypto.clip(abs_sub_result_int32, 0, 1)
    cum_sum_offset = pypto.cumsum(count_of_equals, 1)
    cum_sum_offset_int32 = pypto.cast(cum_sum_offset, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
    expert_offset_result = pypto.sub(cum_sum_offset_int32, calc_index)
    expert_offset_result_fp32 = pypto.cast(expert_offset_result, pypto.DT_FP32, pypto.CastMode.CAST_TRUNC)
    expert_offset_abs_fp32 = pypto.abs(expert_offset_result_fp32)
    expert_offset = pypto.cast(expert_offset_abs_fp32, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
    return expert_offset


def moe_distributed_dispatch_kernel(
    moe_case: MoeCase,
    group_name: str,
) -> Callable[[pypto.Tensor, pypto.Tensor], tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]]:
    batch_size = moe_case.batch_size
    hidden_size = moe_case.hidden_size
    moe_expert_num = moe_case.moe_expert_num
    topk = moe_case.topk
    data_type = moe_case.pypto_data_type
    act_data_type = moe_case.torch_data_type
    world_size = moe_case.world_size

    check_cond(isinstance(group_name, str), f'type of group_name must be str, but got {type(group_name)}')
    check_cond(group_name.strip(), f"group_name can't be empty string")
    check_cond(
        1 <= len(group_name) < 128,
        f'the length of group_name only supports [1, 128), but got {len(group_name)}'
    )
    check_cond(batch_size == 8, f'batch_size must be 8, but got {batch_size}')
    check_cond(hidden_size == 5120, f'hidden_size must be 5120, but got {hidden_size}')
    check_cond(moe_expert_num == 160, f'moe_expert_num must be 160, but got {moe_expert_num}')
    check_cond(topk == 8, f'topk must be 8, but got {topk}')
    check_cond(data_type == pypto.DT_BF16, f'data_type must be pypto.DT_BF16, but got {data_type}')
    check_cond(act_data_type == torch.bfloat16, f'torch_data_type must be torch.bfloat16, but got {act_data_type}')
    check_cond(world_size == 4 or world_size == 8, f'world_size must be 4 or 8, but got {world_size}')

    expand_x_row = min(topk * batch_size * world_size, batch_size * moe_expert_num)
    expert_num_per_rank = moe_expert_num // world_size
    info_size = 64
    cum_sum_row_size = align_up(moe_expert_num, 256)
    count_size = 8

    @pypto.frontend.jit()
    def kernel(
        x: pypto.Tensor([batch_size, hidden_size], data_type, format=pypto.TileOpFormat.TILEOP_ND),
        expert_ids: pypto.Tensor([batch_size, topk], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
    ) -> (
        pypto.Tensor([expand_x_row, hidden_size], data_type),
        pypto.Tensor([expand_x_row, info_size], pypto.DT_INT32),
        pypto.Tensor([expert_num_per_rank], pypto.DT_INT32),
        pypto.Tensor([1], pypto.DT_INT32),
    ):
        expand_x = pypto.Tensor([expand_x_row, hidden_size], x.dtype)
        assist_info_for_combine = pypto.Tensor([expand_x_row, info_size], pypto.DT_INT32)
        expert_token_nums = pypto.Tensor([expert_num_per_rank], pypto.DT_INT32)
        recv_counts = pypto.Tensor([1], pypto.DT_INT32)
        this_rank = pypto.distributed.my_symbolic_pe(group_name)

        # 创建通信共享区域
        shmem_data, shmem_data_signal = pypto.distributed.create_shmem_tensor(
            group_name, world_size, x.dtype, [moe_expert_num, batch_size, hidden_size])
        shmem_info, _ = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_INT32, [moe_expert_num, batch_size, info_size])
        shmem_count, shmem_count_signal = pypto.distributed.create_shmem_tensor(
            group_name, world_size, pypto.DT_INT32, [1, cum_sum_row_size, count_size])

        # 根据专家表计算发送偏移
        pypto.set_vec_tile_shapes(1, batch_size * topk)
        expert_ids_vec = pypto.reshape(expert_ids, [1, batch_size * topk])
        offset_table = pypto.Tensor([batch_size, topk], pypto.DT_INT32)
        for _ in pypto.loop(1, name='MOE_DISTRIBUTED_DISPATCH_PREPARE', idx_name='_'):
            for index in range(1, batch_size * topk):
                row_index, col_index = divmod(index, topk)
                pypto.set_vec_tile_shapes(1, info_size)
                remote_expert_id = expert_ids[row_index, col_index]
                token_offset_result = dispatch_calc_occurrences(expert_ids_vec, remote_expert_id, index)
                token_index = max(index - 1, 0)
                token_offset = token_offset_result[0, token_index]
                offset_table[row_index, col_index] = token_offset

        # 发送 token 与 info 信息
        for _ in pypto.loop(1, name='MOE_DISTRIBUTED_DISPATCH_SEND_DATA', idx_name='_'):
            for index in range(batch_size * topk):
                moe_info = pypto.Tensor([1, info_size], pypto.DT_INT32)
                row_index, col_index = divmod(index, topk)
                tensor_tile = x[row_index:row_index + 1, :]
                pypto.set_vec_tile_shapes(1, info_size)
                moe_info[0, info_size - 3] = this_rank
                moe_info[0, info_size - 2] = row_index
                moe_info[0, info_size - 1] = col_index
                remote_expert_id = expert_ids[row_index, col_index]
                remote_rank_id = remote_expert_id // pypto.SymbolicScalar(expert_num_per_rank)
                remote_expert_offset = remote_expert_id % expert_num_per_rank
                token_offset = offset_table[row_index, col_index]
                pypto.set_vec_tile_shapes(1, hidden_size)
                shmem_data_out_put = pypto.distributed.shmem_put(
                    tensor_tile,
                    [remote_expert_offset * world_size + this_rank, token_offset, 0],
                    shmem_data,
                    remote_rank_id,
                    pred=[offset_table],
                )
                pypto.set_vec_tile_shapes(1, info_size)
                shmem_info_out_put = pypto.distributed.shmem_put(
                    moe_info,
                    [remote_expert_offset * world_size + this_rank, token_offset, 0],
                    shmem_info,
                    remote_rank_id,
                    pred=[offset_table],
                )
                pypto.set_vec_tile_shapes(1, hidden_size)
                pypto.distributed.shmem_signal(
                    shmem_data_signal,
                    world_size,
                    1,
                    [world_size, 1, 1, 1, hidden_size],
                    [0, 0, 0, 0, 0],
                    sig_op=pypto.AtomicType.ADD,
                    pred=[shmem_data_out_put, shmem_info_out_put],
                )

        # 发送每个专家的 token 有效发送数目
        for expert_id in pypto.loop(moe_expert_num, name='MOE_DISTRIBUTED_DISPATCH_SEND_COUNT', idx_name='expert_id'):
            expert_offset = dispatch_calc_occurrences(expert_ids_vec, expert_id, batch_size * topk)
            remote_rank_id = expert_id // pypto.SymbolicScalar(expert_num_per_rank)
            remote_expert_offset = expert_id % expert_num_per_rank
            total_offset_tile = expert_offset[:, -1:]
            pypto.set_vec_tile_shapes(1, 1)
            shmem_put_out = pypto.distributed.shmem_put(
                total_offset_tile,
                [0, remote_expert_offset * world_size + this_rank + 1, 0],
                shmem_count,
                remote_rank_id,
                pred=[total_offset_tile],
            )
            pypto.set_vec_tile_shapes(1, count_size)
            pypto.distributed.shmem_signal(
                shmem_count_signal,
                remote_rank_id,
                1,
                [1, 1, 1, 1, count_size],
                [remote_rank_id, 0, 0, 0, 0],
                sig_op=pypto.AtomicType.ADD, pred=[shmem_put_out],
            )

        # 接收 count 值 计算专家接收数据在输出上的偏移
        cum_sum_result = pypto.tensor([cum_sum_row_size, count_size], pypto.DT_INT32, "cumSumResult")
        local_expert_recv_count = pypto.tensor([cum_sum_row_size, count_size], pypto.DT_INT32, "localExpertRecvCount")
        for _ in pypto.loop(1, name='MOE_DISTRIBUTED_DISPATCH_CUM_SUM', idx_name='_'):
            pypto.set_vec_tile_shapes(1, hidden_size)
            shmem_data_wait_out = pypto.distributed.shmem_wait_until(
                shmem_data_signal,
                pypto.OpType.EQ,
                batch_size * topk * world_size,
                [1, 1, 1, 1, hidden_size],
                [this_rank, 0, 0, 0, 0],
                clear_signal=True,
                pred=[cum_sum_result],
            )
            pypto.set_vec_tile_shapes(1, count_size)
            shmem_count_wait_out = pypto.distributed.shmem_wait_until(
                shmem_count_signal,
                pypto.OpType.EQ,
                moe_expert_num,
                [1, 1, 1, 1, count_size],
                [this_rank, 0, 0, 0, 0],
                clear_signal=True,
                pred=[cum_sum_result],
            )
            pypto.set_vec_tile_shapes(cum_sum_row_size, count_size)
            local_expert_recv_count = pypto.distributed.shmem_get(
                shmem_count,
                this_rank,
                [1, cum_sum_row_size, count_size],
                [0, 0, 0],
                pred=[shmem_data_wait_out, shmem_count_wait_out],
            )
            pypto.set_vec_tile_shapes(cum_sum_row_size, count_size)
            cum_sum_input = pypto.distributed.shmem_get(
                shmem_count,
                this_rank,
                [1, cum_sum_row_size, count_size],
                [0, 0, 0],
                pred=[shmem_data_wait_out, shmem_count_wait_out],
            )
            cum_sum_current = pypto.cumsum(cum_sum_input, 0)
            cum_sum_result = pypto.cast(cum_sum_current, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)

            recv_count_result = cum_sum_result[expert_num_per_rank * world_size, 0]
            recv_counts[0] = recv_count_result
            for expert_id in range(expert_num_per_rank):
                expert_valid_cnt = cum_sum_input[expert_id * world_size + 1:expert_id * world_size + 1 + world_size, :]
                expert_valid_cum_sum = pypto.cumsum(expert_valid_cnt, 0)
                expert_valid_cum_sum_int32 = pypto.cast(expert_valid_cum_sum, pypto.DT_INT32, pypto.CastMode.CAST_TRUNC)
                recv_valid_result = expert_valid_cum_sum_int32[world_size - 1, 0]
                expert_token_nums[expert_id] = recv_valid_result

        # 根据偏移值 做 token 与 info 的数据接收
        for _ in pypto.loop(1, name='MOE_DISTRIBUTED_DISPATCH_RECEIVE', idx_name='_'):
            for index in range(expert_num_per_rank * world_size):
                cur_count = local_expert_recv_count[index + 1, 0]
                offset = cum_sum_result[index, 0]
                pypto.set_vec_tile_shapes(batch_size, hidden_size)
                local_data_recv_count = pypto.experimental.shmem_load(
                    shmem_data,
                    this_rank,
                    [1, batch_size, hidden_size],
                    [index, 0, 0],
                    pred=[cum_sum_result],
                    valid_shape=[1, 1, cur_count, hidden_size],
                )
                expand_x[offset:offset + cur_count, :hidden_size] = local_data_recv_count
                pypto.set_vec_tile_shapes(batch_size, info_size)
                local_info_recv_count = pypto.experimental.shmem_load(
                    shmem_info,
                    this_rank,
                    [1, batch_size, info_size],
                    [index, 0, 0],
                    pred=[cum_sum_result],
                    valid_shape=[1, 1, cur_count, info_size],
                )
                assist_info_for_combine[offset:offset + cur_count, :info_size] = local_info_recv_count

        return expand_x, assist_info_for_combine, expert_token_nums, recv_counts

    return kernel


def test_moe_distributed_dispatch(
    moe_case: MoeCase,
    operand: MoeDispatchOperand,
) -> None:
    (
        x,
        expert_ids,
        assist_info_for_combine,
        expand_x,
        expert_token_nums,
        recv_counts,
        rank_id,
    ) = (
        operand.x,
        operand.expert_ids,
        operand.assist_info_for_combine,
        operand.expand_x,
        operand.expert_token_nums,
        operand.recv_counts,
        operand.rank_id,
    )
    groups = init_hccl_comm(rank_id)
    x.share_memory_()
    expert_ids.share_memory_()
    expand_x.share_memory_()
    assist_info_for_combine.share_memory_()
    expert_token_nums.share_memory_()
    recv_counts.share_memory_()
    rank_id = rank_id + PHYSICAL_START_DEVICE_ID
    x = x.to(f'npu:{rank_id}')
    expert_ids = expert_ids.to(f'npu:{rank_id}')
    expand_x = expand_x.to(f'npu:{rank_id}')
    assist_info_for_combine = assist_info_for_combine.to(f'npu:{rank_id}')
    expert_token_nums = expert_token_nums.to(f'npu:{rank_id}')
    recv_counts = recv_counts.to(f'npu:{rank_id}')
    kernel = moe_distributed_dispatch_kernel(moe_case=moe_case, group_name=groups[0])
    (
        expand_x_kernel,
        assist_info_for_combine_kernel,
        expert_token_nums_kernel,
        recv_counts_kernel,
    ) = kernel(x, expert_ids)

    np.testing.assert_allclose(
        np.array(expand_x_kernel.cpu().flatten().tolist()),
        np.array(expand_x.cpu().flatten().tolist()),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.array(assist_info_for_combine_kernel.cpu().flatten().tolist()),
        np.array(assist_info_for_combine.cpu().flatten().tolist()),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.array(expert_token_nums_kernel.cpu().flatten().tolist()),
        np.array(expert_token_nums.cpu().flatten().tolist()),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        np.array(recv_counts_kernel.cpu().flatten().tolist()),
        np.array(recv_counts.cpu().flatten().tolist()),
        rtol=0,
        atol=0,
    )


def main() -> None:
    mp.set_start_method('spawn', force=True)
    processes = []

    moe_case = MoeCase(
        batch_size=8,
        hidden_size=5120,
        moe_expert_num=160,
        topk=8,
        pypto_data_type=pypto.DT_BF16,
        torch_data_type=torch.bfloat16,
        world_size=WORLD_SIZE,
    )
    dispatch_golden_data = generate_dispatch_golden(moe_case)
    for x, moe_expert_ids, expand_x, assist_info_for_combine, recv_counts, expert_token_nums, rank_id in zip(
        dispatch_golden_data.x_list,
        dispatch_golden_data.moe_expert_ids_list,
        dispatch_golden_data.expand_x_list,
        dispatch_golden_data.assist_info_for_combine_list,
        dispatch_golden_data.recv_counts_list,
        dispatch_golden_data.expert_token_nums_list,
        LOGICAL_RANKS,
    ):
        operand = MoeDispatchOperand(
            x,
            moe_expert_ids,
            expand_x,
            assist_info_for_combine,
            expert_token_nums,
            recv_counts,
            rank_id,
        )
        p = mp.Process(
            target=test_moe_distributed_dispatch,
            args=(
                moe_case,
                operand,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
