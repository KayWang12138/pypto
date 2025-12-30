#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import torch.distributed as dist



MASTER_IP = "127.0.0.1"
MASTER_PORT = "50001"
WORLD_SIZE = 4
RANK_LIST = [0, 1, 2, 3]  # 物理卡号
PHYSICAL_TO_LOGICAL = {0: 0, 1: 1, 2: 2, 3: 3}  # 物理卡号->逻辑映射


class CommMgr:
    def __init__(self, rank):
        self._comm_group = None
        torch_npu.npu.set_device(rank)
        self.logical_rank = PHYSICAL_TO_LOGICAL[rank]
        dist.init_process_group(
        backend="hccl",
        rank=self.logical_rank,
        world_size=WORLD_SIZE,
        init_method=f"tcp://{MASTER_IP}:{MASTER_PORT}"
        )
        logical_ranks = [PHYSICAL_TO_LOGICAL[r] for r in RANK_LIST]
        self.group_handle = dist.new_group(backend="hccl", ranks=logical_ranks)
    
    
    def get_comm_name(self):
        comm_name = self.group_handle._get_backend(torch.device("npu")).get_hccl_comm_name(self.logical_rank)
        self._comm_group_names = []
        self._comm_group_names.append(comm_name)
        return self._comm_group_names


@pypto.jit(
    host_options={"only_codegen": True},
)
def shmem_all_reduce(in_tensor, out_tensor, group):
    in_shape = in_tensor.shape
    pypto.set_dist_tile_shapes([in_shape[0], 1, 0], [in_shape[1], 1, 0], [1, WORLD_SIZE, 0])
    pypto.one_shot_shmem_all_reduce(in_tensor, group, out_tensor)


def generate_golden_data(shape, data_type):
    in_tensors = []
    for _ in range(len(RANK_LIST)):
        in_tensor = torch.randn(shape, dtype=data_type)
        in_tensor.share_memory_()
        in_tensors.append(in_tensor)
    out_tensor_golden = torch.zeros_like(in_tensors[0])
    for tensor in in_tensors:
        out_tensor_golden += tensor
    return [in_tensors, out_tensor_golden]


def verify_result(out_tensor, out_tensor_golden):
    assert_allclose(
        np.array(out_tensor.cpu().flatten().tolist()),
        np.array(out_tensor_golden).flatten(),
        rtol=0.001,
        atol=0.001
    )


def worker(in_tensor, out_tensor_golden, rank):
    comm_mgr = CommMgr(rank)
    groups = comm_mgr.get_comm_name()
    out_tensor = torch.rand((4, 64), dtype=torch.float32, device=f'npu:{rank}')
    pto_inputs = pypto.from_torch(in_tensor.to(f'npu:{rank}')) 
    pto_outputs = pypto.from_torch(out_tensor) 
    shmem_all_reduce(pto_inputs, pto_outputs, groups[0])
    torch_npu.npu.synchronize()
    verify_result(out_tensor, out_tensor_golden)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    processes = []
    in_tensors, out_tensor_golden = generate_golden_data((4, 64), torch.float32)
    for idx, rank in enumerate(RANK_LIST):
        p = mp.Process(target=worker, args=(in_tensors[idx], out_tensor_golden, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()