#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""DistributedConfig for multi-rank Aigcode verification."""

import os

import torch
import torch.distributed as dist
import torch_npu


class DistributedConfig:
    """Communication config for multi-rank Aigcode tests."""

    def __init__(self, world_size: int = 0, master_ip: str = "127.0.0.1"):
        self.master_ip = master_ip
        self.world_size = world_size
        self._parse_device_list()
        self.logical_ranks = list(range(self.world_size))
        self.master_port = self._calculate_port()

    def init_hccl_comm(self, logical_rank_id: int, comm_count: int = 1) -> list[str]:
        physical_device_id = self.get_physical_device_id(logical_rank_id)
        torch_npu.npu.set_device(physical_device_id)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="hccl",
                rank=logical_rank_id,
                world_size=self.world_size,
                init_method=f"tcp://{self.master_ip}:{self.master_port}",
            )
        return [self._get_hccl_comm_name(logical_rank_id) for _ in range(comm_count)]

    def get_physical_device_id(self, logical_rank_id: int) -> int:
        if logical_rank_id >= len(self.physical_device_ids):
            raise ValueError(
                f"Logical rank {logical_rank_id} out of range. "
                f"Available physical devices: {self.physical_device_ids}"
            )
        return self.physical_device_ids[logical_rank_id]

    def _get_hccl_comm_name(self, logical_rank_id: int) -> str:
        group_handle = dist.new_group(backend="hccl", ranks=self.logical_ranks)
        backend = group_handle._get_backend(torch.device("npu"))
        return backend.get_hccl_comm_name(logical_rank_id)

    def _parse_device_list(self) -> None:
        device_list_str = os.environ.get("TILE_FWK_DEVICE_ID_LIST", "")
        if device_list_str:
            self.physical_device_ids = [int(d.strip()) for d in device_list_str.split(",") if d.strip()]
            self.world_size = len(self.physical_device_ids)
        else:
            self.physical_device_ids = list(range(self.world_size))

    def _calculate_port(self) -> int:
        if not self.physical_device_ids:
            return 50001
        port = 5000 + self.physical_device_ids[0]
        if port < 1024 or port > 65535:
            return 50001
        return port
