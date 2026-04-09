#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Aigcode distributed MoE dispatch + combine validation with fixed Qwen3 Next training shape."""

import dataclasses
import multiprocessing as mp

import torch
import torch.distributed as dist

import pypto

from utils.distributed_config import DistributedConfig


AIGCODE_LOCAL_TOKENS = 8192
AIGCODE_CHUNK_TOKENS = 1024
AIGCODE_HIDDEN_SIZE = 4096
AIGCODE_MOE_EXPERT_NUM = 8
AIGCODE_TOPK = 2
AIGCODE_EP_WORLD_SIZE = 8
AIGCODE_EXPERTS_PER_RANK = AIGCODE_MOE_EXPERT_NUM // AIGCODE_EP_WORLD_SIZE
AIGCODE_DISPATCH_ROWS = min(
    AIGCODE_CHUNK_TOKENS * AIGCODE_TOPK * AIGCODE_EP_WORLD_SIZE,
    AIGCODE_CHUNK_TOKENS * AIGCODE_MOE_EXPERT_NUM,
)
AIGCODE_CHUNK_TURNS = AIGCODE_LOCAL_TOKENS // AIGCODE_CHUNK_TOKENS
AIGCODE_TORCH_DTYPE = torch.bfloat16
AIGCODE_PYPTO_DTYPE = pypto.DT_BF16
AIGCODE_COMPILE_DEBUG_MODE = 0
AIGCODE_RUNTIME_DEBUG_MODE = 3


def check_cond(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class MoeCase:
    batch_size: int
    hidden_size: int
    moe_expert_num: int
    topk: int
    ep_world_size: int

    @property
    def experts_per_rank(self) -> int:
        return self.moe_expert_num // self.ep_world_size

    @property
    def dispatch_rows(self) -> int:
        return min(self.batch_size * self.topk * self.ep_world_size, self.batch_size * self.moe_expert_num)


CHUNK_CASE = MoeCase(
    batch_size=AIGCODE_CHUNK_TOKENS,
    hidden_size=AIGCODE_HIDDEN_SIZE,
    moe_expert_num=AIGCODE_MOE_EXPERT_NUM,
    topk=AIGCODE_TOPK,
    ep_world_size=AIGCODE_EP_WORLD_SIZE,
)


def assert_equal_tensor(expected: torch.Tensor, actual: torch.Tensor, name: str) -> None:
    if expected.shape != actual.shape:
        raise AssertionError(f"{name} shape mismatch: expected {expected.shape}, got {actual.shape}")
    if expected.dtype != actual.dtype:
        raise AssertionError(f"{name} dtype mismatch: expected {expected.dtype}, got {actual.dtype}")
    if not torch.equal(expected, actual):
        raise AssertionError(f"{name} mismatch")


def assert_close_tensor(expected: torch.Tensor, actual: torch.Tensor, name: str, atol: float) -> None:
    if expected.shape != actual.shape:
        raise AssertionError(f"{name} shape mismatch: expected {expected.shape}, got {actual.shape}")
    if expected.dtype != actual.dtype:
        raise AssertionError(f"{name} dtype mismatch: expected {expected.dtype}, got {actual.dtype}")
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol, msg=name)


def create_zero_tensor_on_npu(template: torch.Tensor, device_id: int) -> torch.Tensor:
    return torch.zeros(template.shape, dtype=template.dtype, device=f"npu:{device_id}")


def sync_ep_group() -> None:
    dist.barrier()
    torch.npu.synchronize()


def generate_inputs() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(0)
    x_list = []
    expert_ids_list = []
    expert_scales_list = []
    for _ in range(AIGCODE_EP_WORLD_SIZE):
        x = torch.randn((AIGCODE_LOCAL_TOKENS, AIGCODE_HIDDEN_SIZE), dtype=torch.float32, generator=generator)
        x_list.append(x.to(AIGCODE_TORCH_DTYPE))

        expert_scores = torch.randn(
            (AIGCODE_LOCAL_TOKENS, AIGCODE_MOE_EXPERT_NUM), dtype=torch.float32, generator=generator
        )
        topk_scores, expert_ids = expert_scores.topk(k=AIGCODE_TOPK, dim=-1)
        expert_ids_list.append(expert_ids.to(torch.int32))
        expert_scales_list.append(topk_scores.softmax(dim=-1))
    return x_list, expert_ids_list, expert_scales_list


def get_expert_rank_and_offset(expert_id: int, experts_per_rank: int) -> tuple[int, int]:
    return divmod(expert_id, experts_per_rank)


def dispatch_tokens(
    moe_case: MoeCase,
    x_list: list[torch.Tensor],
    expert_ids_list: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    experts_per_rank = moe_case.experts_per_rank
    expand_x_per_expert = [[[] for _ in range(experts_per_rank)] for _ in range(moe_case.ep_world_size)]
    assist_info_per_expert = [[[] for _ in range(experts_per_rank)] for _ in range(moe_case.ep_world_size)]

    for sending_rank_id, (x, expert_ids) in enumerate(zip(x_list, expert_ids_list)):
        for token_id, (token, topk_expert_ids) in enumerate(zip(x, expert_ids)):
            for k_offset, moe_expert_id in enumerate(topk_expert_ids):
                receiving_rank_id, expert_offset = get_expert_rank_and_offset(moe_expert_id.item(), experts_per_rank)
                expand_x_per_expert[receiving_rank_id][expert_offset].append(token)
                assist_info_per_expert[receiving_rank_id][expert_offset].append((sending_rank_id, token_id, k_offset))

    expand_x_per_rank = []
    assist_info_per_rank = []
    expert_token_nums_per_rank = []
    recv_counts_per_rank = []
    for logical_rank_id in range(moe_case.ep_world_size):
        expand_x = torch.zeros((moe_case.dispatch_rows, moe_case.hidden_size), dtype=AIGCODE_TORCH_DTYPE)
        assist_info = torch.zeros((moe_case.dispatch_rows, 3), dtype=torch.int32)
        expert_token_nums = torch.zeros((experts_per_rank,), dtype=torch.int32)
        offset = 0

        for expert_offset in range(experts_per_rank):
            tokens = expand_x_per_expert[logical_rank_id][expert_offset]
            if not tokens:
                continue
            actual_expand_x = torch.stack(tokens, dim=0)
            end = offset + actual_expand_x.size(0)
            expand_x[offset:end] = actual_expand_x
            actual_assist_info = torch.tensor(assist_info_per_expert[logical_rank_id][expert_offset], dtype=torch.int32)
            assist_info[offset:end] = actual_assist_info
            expert_token_nums[expert_offset] = actual_expand_x.size(0)
            offset = end

        expand_x_per_rank.append(expand_x)
        assist_info_per_rank.append(assist_info)
        expert_token_nums_per_rank.append(expert_token_nums)
        recv_counts_per_rank.append(expert_token_nums.sum(dtype=torch.int32).unsqueeze(0))

    return expand_x_per_rank, assist_info_per_rank, expert_token_nums_per_rank, recv_counts_per_rank


def combine_tokens(
    moe_case: MoeCase,
    expand_x_list: list[torch.Tensor],
    assist_info_list: list[torch.Tensor],
    recv_counts_list: list[torch.Tensor],
    expert_scales_list: list[torch.Tensor],
) -> list[torch.Tensor]:
    moe_expert_tokens_list = [
        torch.zeros((moe_case.batch_size, moe_case.topk, moe_case.hidden_size), dtype=AIGCODE_TORCH_DTYPE)
        for _ in range(moe_case.ep_world_size)
    ]

    for expand_x, assist_info, recv_counts in zip(expand_x_list, assist_info_list, recv_counts_list):
        for row_index in range(recv_counts.item()):
            token = expand_x[row_index]
            logical_rank_id, token_id, k_offset = assist_info[row_index]
            moe_expert_tokens_list[logical_rank_id][token_id, k_offset] = token

    out_list = []
    for moe_expert_tokens, expert_scales in zip(moe_expert_tokens_list, expert_scales_list):
        out = (
            expert_scales.unsqueeze(1)
            .matmul(moe_expert_tokens.to(torch.float32))
            .squeeze(1)
            .to(AIGCODE_TORCH_DTYPE)
        )
        out_list.append(out)
    return out_list


def build_dispatch_kernel(group_name: str, runtime_debug_mode: int):
    @pypto.frontend.jit(debug_options={"runtime_debug_mode": runtime_debug_mode})
    def kernel(
        x: pypto.Tensor(
            [AIGCODE_CHUNK_TOKENS, AIGCODE_HIDDEN_SIZE],
            AIGCODE_PYPTO_DTYPE,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
        expert_ids: pypto.Tensor(
            [AIGCODE_CHUNK_TOKENS, AIGCODE_TOPK],
            pypto.DT_INT32,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
        expand_x: pypto.Tensor(
            [AIGCODE_DISPATCH_ROWS, AIGCODE_HIDDEN_SIZE],
            AIGCODE_PYPTO_DTYPE,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
        assist_info_for_combine: pypto.Tensor(
            [AIGCODE_DISPATCH_ROWS, 3],
            pypto.DT_INT32,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
        expert_token_nums: pypto.Tensor(
            [AIGCODE_EXPERTS_PER_RANK],
            pypto.DT_INT32,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
    ):
        pypto.distributed.moe_distributed_dispatch(
            x,
            expert_ids,
            group_name,
            AIGCODE_MOE_EXPERT_NUM,
            AIGCODE_EP_WORLD_SIZE,
            expand_x,
            expert_token_nums,
            assist_info_for_combine,
        )

    return kernel


def build_combine_kernel(group_name: str, runtime_debug_mode: int):
    @pypto.frontend.jit(debug_options={"runtime_debug_mode": runtime_debug_mode})
    def kernel(
        expand_x: pypto.Tensor(
            [AIGCODE_DISPATCH_ROWS, AIGCODE_HIDDEN_SIZE],
            AIGCODE_PYPTO_DTYPE,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
        assist_info_for_combine: pypto.Tensor(
            [AIGCODE_DISPATCH_ROWS, 3],
            pypto.DT_INT32,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
        recv_counts: pypto.Tensor([1], pypto.DT_INT32, format=pypto.TileOpFormat.TILEOP_ND),
        expert_scales: pypto.Tensor(
            [AIGCODE_CHUNK_TOKENS, AIGCODE_TOPK],
            pypto.DT_FP32,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
        out: pypto.Tensor(
            [AIGCODE_CHUNK_TOKENS, AIGCODE_HIDDEN_SIZE],
            AIGCODE_PYPTO_DTYPE,
            format=pypto.TileOpFormat.TILEOP_ND,
        ),
    ):
        pypto.distributed.moe_distributed_combine_v2(
            expand_x,
            assist_info_for_combine,
            recv_counts,
            expert_scales,
            group_name,
            AIGCODE_EP_WORLD_SIZE,
            AIGCODE_MOE_EXPERT_NUM,
            0,
            0,
            out,
        )

    return kernel


def run_aigcode_dispatch_combine(
    config: DistributedConfig,
    x_list: list[torch.Tensor],
    expert_ids_list: list[torch.Tensor],
    expert_scales_list: list[torch.Tensor],
    logical_rank_id: int,
    chunk_turns: int,
    compile_debug_mode: int,
    runtime_debug_mode: int,
) -> None:
    pypto.set_debug_options(compile_debug_mode=compile_debug_mode, runtime_debug_mode=runtime_debug_mode)
    group_name = config.init_hccl_comm(logical_rank_id)[0]
    physical_device_id = config.get_physical_device_id(logical_rank_id)

    dispatch_kernel = build_dispatch_kernel(group_name, runtime_debug_mode)
    combine_kernel = build_combine_kernel(group_name, runtime_debug_mode)

    for turn in range(chunk_turns):
        start = turn * AIGCODE_CHUNK_TOKENS
        end = start + AIGCODE_CHUNK_TOKENS
        x_chunk_list = [x[start:end] for x in x_list]
        expert_ids_chunk_list = [expert_ids[start:end] for expert_ids in expert_ids_list]
        expert_scales_chunk_list = [expert_scales[start:end] for expert_scales in expert_scales_list]

        expand_x_golden_list, assist_info_golden_list, expert_token_nums_golden_list, recv_counts_golden_list = (
            dispatch_tokens(CHUNK_CASE, x_chunk_list, expert_ids_chunk_list)
        )
        out_golden_list = combine_tokens(
            CHUNK_CASE,
            expand_x_golden_list,
            assist_info_golden_list,
            recv_counts_golden_list,
            expert_scales_chunk_list,
        )

        x = x_chunk_list[logical_rank_id].to(f"npu:{physical_device_id}")
        expert_ids = expert_ids_chunk_list[logical_rank_id].to(f"npu:{physical_device_id}")
        expert_scales = expert_scales_chunk_list[logical_rank_id].to(f"npu:{physical_device_id}")

        expand_x_golden = expand_x_golden_list[logical_rank_id]
        assist_info_golden = assist_info_golden_list[logical_rank_id]
        expert_token_nums_golden = expert_token_nums_golden_list[logical_rank_id]
        recv_counts_golden = recv_counts_golden_list[logical_rank_id]
        out_golden = out_golden_list[logical_rank_id]

        expand_x_actual = create_zero_tensor_on_npu(expand_x_golden, physical_device_id)
        assist_info_actual = create_zero_tensor_on_npu(assist_info_golden, physical_device_id)
        expert_token_nums_actual = create_zero_tensor_on_npu(expert_token_nums_golden, physical_device_id)
        out_actual = create_zero_tensor_on_npu(out_golden, physical_device_id)

        dispatch_kernel(
            x,
            expert_ids,
            expand_x_actual,
            assist_info_actual,
            expert_token_nums_actual,
        )
        sync_ep_group()
        recv_counts_actual = expert_token_nums_actual.sum(dtype=torch.int32).reshape(1)
        assert_equal_tensor(expand_x_golden, expand_x_actual.cpu(), f"dispatch.expand_x.turn_{turn}")
        assert_equal_tensor(assist_info_golden, assist_info_actual.cpu(), f"dispatch.assist_info.turn_{turn}")
        assert_equal_tensor(
            expert_token_nums_golden, expert_token_nums_actual.cpu(), f"dispatch.expert_token_nums.turn_{turn}"
        )
        assert_equal_tensor(recv_counts_golden, recv_counts_actual.cpu(), f"dispatch.recv_counts.turn_{turn}")

        combine_kernel(expand_x_actual, assist_info_actual, recv_counts_actual, expert_scales, out_actual)
        sync_ep_group()
        assert_close_tensor(out_golden, out_actual.cpu(), f"combine.out.turn_{turn}", atol=1e-2)

    print(f"rank {logical_rank_id}: SUCCESS")


def share_tensor_list(tensors: list[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.share_memory_()


def main() -> None:
    check_cond(
        AIGCODE_LOCAL_TOKENS % AIGCODE_CHUNK_TOKENS == 0,
        "AIGCODE_LOCAL_TOKENS must be divisible by AIGCODE_CHUNK_TOKENS",
    )
    chunk_turns = AIGCODE_LOCAL_TOKENS // AIGCODE_CHUNK_TOKENS
    compile_debug_mode = AIGCODE_COMPILE_DEBUG_MODE
    runtime_debug_mode = AIGCODE_RUNTIME_DEBUG_MODE
    config = DistributedConfig(world_size=AIGCODE_EP_WORLD_SIZE)
    check_cond(
        config.world_size == AIGCODE_EP_WORLD_SIZE,
        f"Aigcode dispatch+combine requires {AIGCODE_EP_WORLD_SIZE} ranks, but got {config.world_size}",
    )

    x_list, expert_ids_list, expert_scales_list = generate_inputs()
    share_tensor_list(x_list)
    share_tensor_list(expert_ids_list)
    share_tensor_list(expert_scales_list)

    mp.set_start_method("spawn", force=True)
    processes = []
    for logical_rank_id in config.logical_ranks:
        process = mp.Process(
            target=run_aigcode_dispatch_combine,
            args=(
                config,
                x_list,
                expert_ids_list,
                expert_scales_list,
                logical_rank_id,
                chunk_turns,
                compile_debug_mode,
                runtime_debug_mode,
            ),
        )
        process.start()
        processes.append(process)

    for logical_rank_id, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            raise AssertionError(f"rank {logical_rank_id} failed with exit code {process.exitcode}")


if __name__ == "__main__":
    main()
