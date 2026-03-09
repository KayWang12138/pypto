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
    """多卡测试配置类"""

    def __init__(
        self,
        world_size: int = 0,
        master_ip: str = "127.0.0.1",
        master_port: int = 50001,
    ):
        """
        初始化多卡测试配置

        Args:
            world_size: 需要的卡数，0表示从环境变量解析
            master_ip: 主节点IP地址
            master_port: 主节点端口
        """
        self.master_ip = master_ip
        self.master_port = master_port

        # 初始化属性
        self.physical_device_ids: List[int] = []
        self.world_size: int = world_size
        self.logical_ranks = list(range(self.world_size))

        # 解析设备列表
        self._parse_device_list()

    def _parse_device_list(self):
        """解析设备列表"""
        device_list_str = os.environ.get("TILE_FWK_DEVICE_ID_LIST", "")

        if device_list_str:
            # 从环境变量解析设备列表
            self.physical_device_ids = [
                int(d.strip()) for d in device_list_str.split(",") if d.strip()
            ]

            # 从设备列表获取卡数
            self.world_size = len(self.physical_device_ids)

            print(f"从环境变量解析得到 {self.world_size} 张卡: {self.physical_device_ids}")
        else:
            print(f"使用默认映射: {self.world_size} 张卡 -> 设备: {self.physical_device_ids}")
            # 逻辑rank列表
            self.physical_device_ids = list(range(self.world_size))

    def get_physical_device_id(self, logical_rank: int) -> int:
        """
        根据逻辑rank获取物理设备ID

        如果没有环境变量，返回0-n的映射

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
