#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import dataclasses
import logging
import pathlib
from typing import List, Tuple

import numpy as np
import torch

from golden_register import GoldenRegister

np.random.seed(0)
torch.manual_seed(0)

DTYPE_STR_TO_TORCH = {
    'bool': torch.bool,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'float16': torch.float16,
    'float32': torch.float,
    'bfloat16': torch.bfloat16,
}

TORCH_DTYPE_TO_NUM = {
    torch.bool: 15,
    torch.uint8: 11,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.float16: 6,
    torch.float: 7,
    torch.bfloat16: 8,
}


@dataclasses.dataclass(frozen=True)
class BaseCase:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    rank_size: int


@dataclasses.dataclass
class MoeCase:
    dtype: torch.dtype
    batch_size: int
    hidden_size: int
    share_expert_num: int
    routing_expert_num: int
    top_k: int
    rank_size: int

    def __post_init__(self):
        if self.share_expert_num + self.routing_expert_num != self.rank_size:
            raise ValueError(
                '(share_expert_num + routing_expert_num) != rank_size: '
                f'{self.share_expert_num} + {self.routing_expert_num} != {self.rank_size}'
            )


@dataclasses.dataclass
class AllGatherAttnPostReducescatterCase:
    dtype: torch.dtype
    batch_size: int
    seq_len: int
    num_heads: int
    kv_lora_rank: int
    value_head_dim: int
    output_hidden_size: int
    rank_size: int


def get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in DTYPE_STR_TO_TORCH:
        raise ValueError(f'Unsupported dtype: {dtype_str}')
    return DTYPE_STR_TO_TORCH[dtype_str]


def get_dtype_num(dtype: torch.dtype) -> int:
    if dtype not in TORCH_DTYPE_TO_NUM:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return TORCH_DTYPE_TO_NUM[dtype]


def parse_base_case(case_name: str, dim: int) -> BaseCase:
    parts = case_name.split('_')
    if len(parts) < dim + 2:
        raise ValueError(f'case_name {case_name} format is error.')
    rank_size = int(parts[-1])
    shape = tuple(map(int, parts[-(dim + 1):-1]))
    dtype = get_dtype(parts[-(dim + 2)])

    case = BaseCase(dtype=dtype, shape=shape, rank_size=rank_size)
    logging.info(f'Case {case_name}, case info: {case}')
    return case


def validate_rank_size(rank_size: int) -> None:
    if rank_size <= 1:
        raise ValueError(f'rank_size must be greater than 1, got {rank_size}')


def save_params(params: Tuple[int], save_dir: pathlib.Path) -> None:
    params_tensor = torch.tensor(params, dtype=torch.int64)
    params_ndarray = params_tensor.numpy()
    params_ndarray.tofile(save_dir / 'params.bin')


def save_tensor(tensor: torch.Tensor, save_path: pathlib.Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)  # 仅改变 tensor 的 dtype 解释方式，内存布局不变
    tensor.numpy().tofile(save_path)


def save_tensor_list(tensors: List[torch.Tensor], save_dir: pathlib.Path, filename_prefix: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for rank, tensor in enumerate(tensors):
        save_tensor(tensor, save_dir / f'{filename_prefix}_rank_{rank}.bin')


def gen_random_tensor(shape: Tuple[int], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.int32 or dtype == torch.int16 or dtype == torch.int8:
        return torch.randint(-10, 10, shape, dtype=dtype)
    else:
        return torch.randn(shape, dtype=dtype)


def gen_random_tensor_list(
    shape: Tuple[int], dtype: torch.dtype, rank_size: int, save_dir: pathlib.Path, filename_prefix: str,
) -> List[torch.Tensor]:
    tensor_list = [gen_random_tensor(shape, dtype) for rank in range(rank_size)]
    save_tensor_list(tensor_list, save_dir, filename_prefix)
    return tensor_list


def all_gather_and_save(
    inputs: List[torch.Tensor], rank_size: int, save_dir: pathlib.Path, filename_prefix: str,
) -> torch.Tensor:
    gathered_output = torch.cat(inputs, dim=0)
    outputs = [gathered_output] * rank_size
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def reduce_scatter_and_save(
    inputs: List[torch.Tensor], row: int, rank_size: int, save_dir: pathlib.Path, filename_prefix: str,
) -> torch.Tensor:
    stacked_output = torch.stack(inputs, dim=0)
    reduced_output = torch.sum(stacked_output, dim=0).to(inputs[0].dtype)
    row_per_rank = row // rank_size
    outputs = [reduced_output[rank * row_per_rank: (rank + 1) * row_per_rank] for rank in range(rank_size)]
    save_tensor_list(outputs, save_dir, filename_prefix)
    return outputs


def generate_all_gather_golden(case_name: str, save_dir: pathlib.Path):
    dim = 2
    case = parse_base_case(case_name, dim)
    row, col = case.shape
    rank_size, dtype = case.rank_size, case.dtype

    validate_rank_size(rank_size)

    params = (row, col, get_dtype_num(dtype))
    save_params(params, save_dir)

    inputs = gen_random_tensor_list((row, col), dtype, rank_size, save_dir, 'input')

    all_gather_and_save(inputs, rank_size, save_dir, 'output')


def generate_reduce_scatter_golden(case_name: str, save_dir: pathlib.Path):
    dim = 2
    case = parse_base_case(case_name, dim)
    row, col = case.shape
    rank_size, dtype = case.rank_size, case.dtype

    validate_rank_size(rank_size)
    if row % rank_size != 0:
        raise ValueError(
            'The first dimension of the input tensor must be an integer multiple of the rank size, '
            f'got row={row}, rank_size={rank_size}'
        )

    params = (row, col, get_dtype_num(dtype))
    save_params(params, save_dir)

    inputs = gen_random_tensor_list((row, col), dtype, rank_size, save_dir, 'input')

    reduce_scatter_and_save(inputs, row, rank_size, save_dir, 'output')


def gen_golden_input_data(case: MoeCase, save_dir: pathlib.Path) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    x_list = []
    expert_ids_list = []
    for rank in range(case.rank_size):
        x = gen_random_tensor((case.batch_size, case.hidden_size), case.dtype)
        x_list.append(x)
        save_tensor(x, save_dir / f'x_rank_{rank}.bin')

        scores = torch.sigmoid(gen_random_tensor((case.batch_size, case.routing_expert_num), torch.float32))
        _, expert_ids = torch.topk(scores, k=case.top_k)
        expert_ids_list.append(expert_ids)

        scales = scores.gather(1, expert_ids)
        save_tensor(scales, save_dir / f'scale_rank_{rank}.bin')

        expert_ids_rank = expert_ids + case.share_expert_num
        expert_ids_rank = expert_ids_rank.to(dtype=torch.int32)
        save_tensor(expert_ids_rank, save_dir / f'expert_ids_rank_{rank}.bin')

    return x_list, expert_ids_list


def gen_golden_data_sended_to_single_share_expert(
        rank_id: int,
        case: MoeCase,
        x_list: List[torch.Tensor],
        save_dir: pathlib.Path,
    ) -> None:
    y = torch.zeros([case.batch_size * case.rank_size, case.hidden_size], dtype=case.dtype)
    combine_info = torch.full([case.batch_size * case.rank_size, 3], -1, dtype=torch.int32)

    # 发给共享专家的kOffset暂时固定为topK
    # 本卡共享专家自己的数据
    offset = 0
    y[offset:offset + case.batch_size] = x_list[rank_id]
    for token_id in range(case.batch_size):
        combine_info[offset + token_id] = torch.tensor([rank_id, token_id, case.top_k], dtype=torch.int32)
    offset += case.batch_size

    # 路由专家发送给共享专家，根据 rank_id = expert_id // share_capacity
    for expert_id in range(case.routing_expert_num):
        routing_expert_rank_id = expert_id + case.share_expert_num
        # 一个共享专家负责MoE专家的容量
        share_capacity = case.routing_expert_num // case.share_expert_num if case.share_expert_num > 0 else 0
        if expert_id // share_capacity == rank_id:
            y[offset:offset + case.batch_size] = x_list[routing_expert_rank_id]
            for token_id in range(case.batch_size):
                combine_info[offset + token_id] = torch.tensor(
                    [routing_expert_rank_id, token_id, case.top_k], dtype=torch.int32,
                )
            offset += case.batch_size

    save_tensor(y, save_dir / f'y_rank_{rank_id}.bin')
    save_tensor(combine_info, save_dir / f'combine_info_rank_{rank_id}.bin')

    valid_count = torch.zeros([128], dtype=torch.int32)
    valid_count[0] = offset
    save_tensor(valid_count, save_dir / f'valid_count_rank_{rank_id}.bin')


def gen_golden_data_sended_to_all_share_experts(
        case: MoeCase,
        x_list: List[torch.Tensor],
        save_dir: pathlib.Path,
    ) -> None:
    for rank_id in range(case.share_expert_num):
        gen_golden_data_sended_to_single_share_expert(rank_id, case, x_list, save_dir)


def gen_golden_data_sended_to_single_routing_expert(
        expert_id: int,
        case: MoeCase,
        x_list: List[torch.Tensor],
        expert_ids_list: List[torch.Tensor],
        save_dir: pathlib.Path,
    ) -> None:
    y = torch.zeros([case.batch_size * case.rank_size, case.hidden_size], dtype=case.dtype)
    combine_info = torch.full([case.batch_size * case.rank_size, 3], -1, dtype=torch.int32)
    offset = 0
    # 所有卡发送给本卡MoE专家
    for rank_id in range(case.rank_size):
        x = x_list[rank_id]
        expert_ids = expert_ids_list[rank_id]
        token_ids, k_offsets = torch.where(expert_ids == expert_id)
        if len(token_ids) > 0:
            y[offset:offset + len(token_ids)] = x[token_ids]
            for i, (token_id, k_offset) in enumerate(zip(token_ids, k_offsets)):
                combine_info[offset + i] = torch.tensor(
                    [rank_id, token_id.item(), k_offset.item()], dtype=torch.int32,
                )
        offset += len(token_ids)
    # MoE专家所在卡的位置
    routing_expert_rank_id = case.share_expert_num + expert_id
    save_tensor(y, save_dir / f'y_rank_{routing_expert_rank_id}.bin')
    save_tensor(combine_info, save_dir / f'combine_info_rank_{routing_expert_rank_id}.bin')
    # 有效的token数量 = 偏移量 offset
    valid_count = torch.zeros([128], dtype=torch.int32)
    valid_count[0] = offset
    save_tensor(valid_count, save_dir / f'valid_count_rank_{routing_expert_rank_id}.bin')


def gen_golden_data_sended_to_all_routing_experts(
        case: MoeCase,
        x_list: List[torch.Tensor],
        expert_ids_list: List[torch.Tensor],
        save_dir: pathlib.Path,
    ) -> None:
    for expert_id in range(case.routing_expert_num):
        gen_golden_data_sended_to_single_routing_expert(
            expert_id, case, x_list, expert_ids_list, save_dir,
        )


def gen_moe_dispatch_case(case: MoeCase, save_dir: pathlib.Path) -> None:
    params = (case.batch_size, case.hidden_size, case.share_expert_num, case.routing_expert_num, case.top_k,
        get_dtype_num(case.dtype))
    save_params(params, save_dir)
    x_list, expert_ids_list = gen_golden_input_data(case, save_dir)
    gen_golden_data_sended_to_all_share_experts(case, x_list, save_dir)
    gen_golden_data_sended_to_all_routing_experts(case, x_list, expert_ids_list, save_dir)


def generate_moe_dispatch_golden(case_name: str, save_dir: pathlib.Path):
    case = MoeCase(
        dtype=torch.bfloat16,
        batch_size=8,
        hidden_size=7168,
        share_expert_num=1,
        routing_expert_num=3,
        top_k=2,
        rank_size=int(case_name.split('_')[-1]),
    )
    gen_moe_dispatch_case(case, save_dir)


def gen_combine_case(case: MoeCase, save_dir: pathlib.Path, dispatch_save_dir: pathlib.Path) -> None:
    # 一个共享专家负责MoE专家的容量
    share_capacity = case.routing_expert_num // case.share_expert_num

    x_list = []
    combine_info_list = []
    scale_list = []
    for rank in range(case.rank_size):
        x = torch.from_numpy(np.fromfile(dispatch_save_dir / f'y_rank_{rank}.bin'))
        x = x.view(dtype=case.dtype).view([-1, case.hidden_size])
        x_list.append(x)

        combine_info = torch.from_numpy(
            np.fromfile(dispatch_save_dir / f'combine_info_rank_{rank}.bin', dtype=np.int32),
        )
        combine_info = combine_info.view([-1, 3])
        combine_info_list.append(combine_info)

        scale = torch.from_numpy(np.fromfile(dispatch_save_dir / f'scale_rank_{rank}.bin', dtype=np.float32))
        scale = scale.view([case.batch_size, -1, 1])
        scale_list.append(scale)

    for rank in range(case.rank_size):
        # Combine共享专家的数据
        share_y = torch.zeros([case.batch_size, case.hidden_size], dtype=case.dtype)
        if case.share_expert_num > 0:
            # 本卡就是共享专家
            if rank < case.share_expert_num:
                rank_share = rank
            else:       # 本卡是MoE专家，根据 rank_share = expert_id // share_capacity 获取专家卡
                rank_share = (rank - case.share_expert_num) // share_capacity
            x = x_list[rank_share]
            combine_info = combine_info_list[rank_share]

            mask = combine_info[:, 0] == rank
            x = x[mask]
            assert x.shape == share_y.shape, 'combine share data error: x.shape != share_y.shape'
            share_y = x
        save_tensor(share_y, save_dir / f'share_y_rank_{rank}.bin')

        # Combine MoE专家的数据
        moe_y = torch.zeros([case.batch_size, case.top_k, case.hidden_size], dtype=case.dtype)
        for expert_id in range(case.routing_expert_num):
            rank_moe = case.share_expert_num + expert_id
            x = x_list[rank_moe]
            combine_info = combine_info_list[rank_moe]

            mask = combine_info[:, 0] == rank
            x = x[mask]
            combine_info = combine_info[mask]

            for idx in range(len(combine_info)):
                token_id = combine_info[idx, 1]
                k_offset = combine_info[idx, 2]
                moe_y[token_id, k_offset] = x[idx]
        save_tensor(moe_y, save_dir / f'moe_y_rank_{rank}.bin')

        # 计算的时候转化为fp32
        scale = scale_list[rank]
        moe_y = moe_y.to(dtype=torch.float32)
        moe_y = moe_y * scale
        save_tensor(moe_y, save_dir / f'scaled_moe_y_rank_{rank}.bin')

        y = torch.sum(moe_y, dim=1) + share_y.to(dtype=torch.float32)
        save_tensor(y.to(dtype=case.dtype), save_dir / f'y_rank_{rank}.bin')


def generate_moe_combine_golden(case_name: str, save_dir: pathlib.Path):
    batch_size = 8
    hidden_size = 7168
    share_expert_num = 1
    routing_expert_num = 3
    top_k = 2
    rank_size = 4
    dtype = torch.bfloat16

    params = (batch_size, hidden_size, top_k, get_dtype_num(dtype))
    save_params(params, save_dir)

    dispatch_case = MoeCase(
        dtype=dtype,
        batch_size=batch_size,
        hidden_size=hidden_size,
        share_expert_num=share_expert_num,
        routing_expert_num=routing_expert_num,
        top_k=top_k,
        rank_size=rank_size,
    )
    dispatch_save_dir = save_dir / 'dispatch'
    dispatch_save_dir.mkdir(parents=True, exist_ok=True)
    gen_moe_dispatch_case(dispatch_case, dispatch_save_dir)

    combine_case = MoeCase(
        dtype=torch.bfloat16,
        batch_size=batch_size,
        hidden_size=hidden_size,
        share_expert_num=share_expert_num,
        routing_expert_num=routing_expert_num,
        top_k=top_k,
        rank_size=rank_size,
    )
    gen_combine_case(combine_case, save_dir, dispatch_save_dir)


def generate_allgather_matmul_reducescatter_golden(case_name: str, save_dir: pathlib.Path):
    dim = 2
    case = parse_base_case(case_name, dim)
    row, col = case.shape
    rank_size, dtype = case.rank_size, case.dtype

    validate_rank_size(rank_size)

    params = (row, col, get_dtype_num(dtype))
    save_params(params, save_dir)
    
    all_gather_inputs = gen_random_tensor_list((row, col), dtype, rank_size, save_dir, 'input')

    all_gather_outputs = all_gather_and_save(all_gather_inputs, rank_size, save_dir, 'allgather')

    gen_random_tensor_list((col, col), dtype, rank_size, save_dir, 'matmul')  # 暂时没用到

    add_output = all_gather_outputs[0] + all_gather_outputs[0]
    add_outputs = []
    for rank in range(rank_size):
        save_tensor(add_output, save_dir / f'ag_add_rank_{rank}.bin')
        add_outputs.append(add_output)

    reduce_scatter_outputs = reduce_scatter_and_save(add_outputs, row * rank_size, rank_size, save_dir, 'rs')

    all_gather_and_save(reduce_scatter_outputs, rank_size, save_dir, 'double_allgather')


def gen_allgather_attnpost_reducescatter_case(case: AllGatherAttnPostReducescatterCase, save_dir: pathlib.Path) -> None:
    batch_size = case.batch_size
    seq_len = case.seq_len
    num_heads = case.num_heads
    kv_lora_rank = case.kv_lora_rank
    value_head_dim = case.value_head_dim
    output_hidden_size = case.output_hidden_size
    rank_size = case.rank_size
    dtype = case.dtype
    params = (
        batch_size,
        seq_len,
        num_heads,
        kv_lora_rank,
        value_head_dim,
        output_hidden_size,
        get_dtype_num(dtype),
    )
    save_params(params, save_dir)

    all_gather_input_shape = (batch_size * seq_len * num_heads // rank_size, kv_lora_rank)
    all_gather_inputs = gen_random_tensor_list(all_gather_input_shape, dtype, rank_size, save_dir, 'ag_in')

    attention_input = torch.cat(all_gather_inputs, dim=0)
    attention_input = attention_input.reshape([batch_size, num_heads, seq_len, kv_lora_rank])
    attention_input = torch.transpose(attention_input, 1, 2)
    attention_input = torch.reshape(attention_input, [batch_size * seq_len, num_heads, kv_lora_rank])
    attention_input = torch.transpose(attention_input, 0, 1)

    reduce_scatter_inputs = []
    for rank in range(rank_size):
        lora_weight = gen_random_tensor((num_heads, kv_lora_rank, value_head_dim), dtype)
        save_tensor(lora_weight, save_dir / f'w_lora_rank_{rank}.bin')

        attention_output = torch.bmm(attention_input.to(torch.float32), lora_weight.to(torch.float32)).to(dtype=dtype)
        attention_output = torch.transpose(attention_output, 0, 1)
        attention_output = torch.reshape(attention_output, [batch_size * seq_len, num_heads * value_head_dim])

        output_weight = gen_random_tensor((num_heads * value_head_dim, output_hidden_size), dtype)
        save_tensor(output_weight, save_dir / f'w_out_rank_{rank}.bin')

        attention_output = torch.matmul(
            attention_output.to(dtype=torch.float32), output_weight.to(dtype=torch.float32)
        ).to(dtype=dtype)
        reduce_scatter_inputs.append(attention_output)

    reduce_scatter_output = torch.stack(reduce_scatter_inputs, dim=0).to(torch.float32)
    reduce_scatter_output = torch.sum(reduce_scatter_output, dim=0).to(dtype)
    batch_per_rank = batch_size * seq_len // rank_size
    for rank in range(rank_size):
        rank_reduce_scatter_output = reduce_scatter_output[rank * batch_per_rank: (rank + 1) * batch_per_rank]
        save_tensor(rank_reduce_scatter_output, save_dir / f'rs_out_rank_{rank}.bin')


def generate_allgather_attn_post_reducescatter_golden(case_name: str, save_dir: pathlib.Path) -> None:
    case = AllGatherAttnPostReducescatterCase(
        batch_size=64,
        seq_len=1,
        num_heads=32,
        kv_lora_rank=256,
        value_head_dim=128,
        output_hidden_size=128,
        rank_size=4,
        dtype=torch.bfloat16,
    )
    gen_allgather_attnpost_reducescatter_case(case, save_dir)


OPERATOR_DISPATCHERS = [
    ('all_gather', generate_all_gather_golden),
    ('reduce_scatter', generate_reduce_scatter_golden),
    ('moe_dispatch', generate_moe_dispatch_golden),
    ('moe_combine', generate_moe_combine_golden),
    ('allgather_matmul_reducescatter', generate_allgather_matmul_reducescatter_golden),
    ('allgather_attn_post_reducescatter', generate_allgather_attn_post_reducescatter_golden),
]


@GoldenRegister.reg_golden_func(
    case_names=[
        'DistributedTest.aicpuWaitFlag_single_test_all_gather_bfloat16_256_256_4',
        'DistributedTest.aicpuWaitFlag_multi_test_all_gather_float16_32_32_4',
        'DistributedTest.aivWaitFlag_single_test_all_gather_bfloat16_256_128_4',
        'DistributedTest.aivWaitFlag_multi_test_all_gather_float16_32_32_4',
        'DistributedTest.shmem_all_gather_int32_128_256_4',
        'DistributedTest.aicpuWaitFlag_single_test_reduce_scatter_int32_32_32_4',
        'DistributedTest.aicpuWaitFlag_multi_test_reduce_scatter_float32_128_256_4',
        'DistributedTest.aivWaitFlag_single_test_reduce_scatter_int32_128_256_4',
        'DistributedTest.aivWaitFlag_multi_test_reduce_scatter_float32_128_256_4',
        'DistributedTest.shmem_reduce_scatter_int32_128_256_4',
        'DistributedTest.aivWaitFlag_single_test_moe_dispatch_bfloat16_rank_size_4',
        'DistributedTest.aivWaitFlag_single_test_moe_combine_bfloat16_rank_size_4',
        'DistributedTest.allgather_attn_post_reducescatter_b64_s1_n32_lora256_dim128_h128_rank4_bf16',
        'DistributedTest.shmem_allgather_matmul_reducescatter_int32_128_256_4',
    ]
)
def generate_golden_case(case_name: str, output: pathlib.Path) -> bool:
    handler = None
    for keyword, func in OPERATOR_DISPATCHERS:
        if keyword in case_name:
            handler = func
            break

    if handler is None:
        raise ValueError(f"Can't find handler for case {case_name}")

    handler(case_name, output)
    logging.info('Generate golden success for %s', case_name)
    return True
