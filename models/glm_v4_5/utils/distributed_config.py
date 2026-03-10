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
DistributedConfig
"""
import os
from typing import List


class DistributedConfig:
    """通信测试配置类"""

    def __init__(
        self,
        world_size: int = 0,
        master_ip: str = "127.0.0.1",
        master_port: int = 50001,
    ):
        """
        初始化多卡测试配置
        Args:
            world_size: 需要的卡数 0表示从环境变量解析
            master_ip: 主节点IP地址
            master_port: 主节点端口
        """
        self.master_ip = master_ip
        self.physical_device_ids: List[int] = []
        self.world_size: int = world_size
        self.logical_ranks = list(range(self.world_size))
        self._parse_device_list()
        self.master_port = master_port

    def _parse_device_list(self):
        """解析设备列表"""
        device_list_str = os.environ.get("TILE_FWK_DEVICE_ID_LIST", "")

        if device_list_str:
            self.physical_device_ids = [
                int(d.strip()) for d in device_list_str.split(",") if d.strip()
            ]
            self.world_size = len(self.physical_device_ids)
        else:
            self.physical_device_ids = list(range(self.world_size))

    def _calculate_port(self) -> int:
        """
        计算端口号
        策略: 5000 + 物理设备ID列表的第一个设备ID
        例如: 设备[4,5] -> 端口 5004
             设备[0,1] -> 端口 5000
             设备[8,9] -> 端口 5008
        返回:
            int: 计算的端口号
        """
        if not self.physical_device_ids:
            return 50001

        first_device_id = self.physical_device_ids[0]
        port = 5000 + first_device_id
        if port < 1024 or port > 65535:
            return 50001

        return port

    def get_physical_device_id(self, logical_rank: int) -> int:
        """
        根据逻辑rank获取物理设备ID
        如果没有环境变量 返回0-n的映射
        Args:
            logical_rank: 逻辑rank
        Returns:
            int: 物理设备ID
        """
        if logical_rank >= len(self.physical_device_ids):
            raise ValueError(
                f"Logical rank {logical_rank} out of range. "
                f"Available physical devices: {self.physical_device_ids}"
            )
        return self.physical_device_ids[logical_rank]
