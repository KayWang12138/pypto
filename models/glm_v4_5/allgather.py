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
AllGather Operation Test Case

This module implements an AllGather operation using shared memory communication.
AllGather gathers data from all ranks and makes it available on every rank.

Main Functions:
    - allgather_kernel: JIT compiled allgather kernel
    - test_allgather: Test function for allgather operation
"""

import multiprocessing as mp

import numpy as np
import pytest
import torch

import pypto

from utils.distributed_config import DistributedConfig


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128,
                     "stitch_cfgcache_size": 100000000},
)
def allgather_kernel(
    in_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    out_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    group_name: str,
    world_size: int,
):
    row = in_tensor.shape[0]
    col = in_tensor.shape[1]

    my_pe = pypto.distributed.my_symbolic_pe(group_name)

    shmem_shape = [world_size * row, col]
    shmem_tensor = pypto.distributed.create_shmem_tensor(
        group_name, world_size, in_tensor.dtype, shmem_shape)

    for dyn_idx in pypto.loop(world_size, name="ALLGATHER_LOOP", idx_name="dyn_idx"):
        pypto.set_vec_tile_shapes(row, col)
        shmem_put_out = pypto.distributed.shmem_put(
            in_tensor, [my_pe * row, 0], shmem_tensor, dyn_idx,
            put_op=pypto.AtomicType.SET, pred=[in_tensor])
        shmem_signal_out = pypto.distributed.shmem_signal(
            shmem_tensor, dyn_idx, 1, [row, col],
            [my_pe * row, 0], target_pe=dyn_idx, sig_op=pypto.AtomicType.SET, pred=[shmem_put_out])

        wait_until_out = pypto.distributed.shmem_wait_until(
            shmem_tensor, my_pe, 1, [row, col],
            [dyn_idx * row, 0], cmp=pypto.OpType.EQ, clear_signal=True, pred=[shmem_signal_out])
        shmem_get_out = pypto.experimental.shmem_load(
            shmem_tensor, my_pe, [row, col], [dyn_idx * row, 0], pred=[wait_until_out], valid_shape=[row, col]
        )
        out_tensor[dyn_idx * row:dyn_idx * row + row, :] = shmem_get_out


def generate_golden_data(world_size: int):
    batch_size = 8
    hidden_size = 5120
    torch.manual_seed(42)

    input_datas = []
    for _ in range(world_size):
        in_tensor = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16).share_memory_()
        input_datas.append(in_tensor)

    output_datas = []
    allgather_result = torch.cat(input_datas, dim=0)
    for _ in range(world_size):
        output_datas.append(allgather_result)

    return input_datas, output_datas


def allgather_worker(
    config: DistributedConfig,
    input_data: torch.Tensor,
    output_data: torch.Tensor,
    logical_rank_id: int,
):
    groups = config.init_hccl_comm(logical_rank_id)
    physical_device_id = config.get_physical_device_id(logical_rank_id)
    device = f'npu:{physical_device_id}'

    in_tensor = input_data.to(device)
    out_tensor = torch.empty(output_data.shape, dtype=torch.bfloat16, device=device)

    allgather_kernel(in_tensor, out_tensor, groups[0], config.world_size)

    np.testing.assert_allclose(
        np.array(out_tensor.cpu().flatten().tolist()),
        np.array(output_data.cpu().flatten().tolist()),
        rtol=8e-3,
        atol=8e-3,
    )


@pytest.mark.world_size(4)
def test_allgather():
    mp.set_start_method('spawn', force=True)
    config = DistributedConfig(world_size=4)
    processes = []
    input_datas, output_datas = generate_golden_data(config.world_size)
    for i in range(config.world_size):
        p = mp.Process(target=allgather_worker, args=(config, input_datas[i], output_datas[i], i))
        p.start()
        processes.append(p)
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise AssertionError(f"process {i} failed, return: {p.exitcode}")


def main():
    test_allgather()


if __name__ == '__main__':
    main()
